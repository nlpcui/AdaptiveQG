import copy

import torch, logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration


class PostionalEncoding(nn.Module):
    def __init__(self, d_model, device='cpu', max_seq_len=5000):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_seq_len: max sequence length
        """
        super(PostionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_seq_len, d_model, requires_grad=False, device=device)

        position = torch.arange(0, max_seq_len, device=device).float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # x: [batch_size, seq_len]
        return self.encoding[:x.size(1), :]


class TransformerEncodingBlock(nn.Module):
    def __init__(self, dim_attn, num_att_heads, dim_ff, dropout):
        super(TransformerEncodingBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(dim_attn, num_att_heads, batch_first=True)
        self.linear_1 = nn.Linear(dim_attn, dim_ff, bias=True)
        self.linear_2 = nn.Linear(dim_ff, dim_attn, bias=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(dim_attn)

    def forward(self, query, key, value, attn_mask):
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, attn_mask=attn_mask) # multi-head attn
        attn_output = self.dropout_1(attn_output)
        attn_output = self.layer_norm(attn_output+query) # residual & layer norm

        ff_output = self.linear_2(self.relu(self.linear_1(attn_output))) # feedforward
        ff_output = self.dropout_2(ff_output)
        ff_output = self.layer_norm(ff_output+query) # residual & layer norm

        return ff_output, attn_output_weights


class DKT(nn.Module):
    def __init__(self, input_size, num_layers, num_users, hidden_size, num_tuples, dropout, device, num_words, num_attn_heads=4, num_labels=3, max_length=1024, encoder='rnn'):
        super(DKT, self).__init__()
        self.device = device
        self.encoder_type = encoder
        if encoder == 'rnn':
            self.encoder = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=True)
        elif encoder == 'transformer':
            self.encoder = nn.ModuleList([TransformerEncodingBlock(input_size, num_attn_heads, hidden_size, dropout) for i in range(num_layers)])

        self.user_embeddings = nn.Embedding(num_users, input_size, device=self.device)
        self.word_embeddings = nn.Embedding(num_words, input_size, padding_idx=0, device=self.device)
        self.label_embeddings = nn.Embedding(num_labels, input_size, padding_idx=2, device=self.device)
        self.tuple_embeddings = nn.Embedding(num_tuples, input_size, padding_idx=0, device=self.device)
        self.position_embeddings = PostionalEncoding(d_model=input_size, device=self.device)
        self.ffn = torch.nn.Linear(hidden_size, num_words)

    def forward(self, x_word_ids, y_labels, x_user_ids):
        # tuple_embeddings = self.tuple_embeddings(x_w_l_tuple_ids)  # batch_size, seq_len, hidden_size
        # input_embeddings = tuple_embeddings
        y_labels[y_labels == -100] = 2
        word_embeddings = self.word_embeddings(x_word_ids)
        label_embeddings = self.label_embeddings(y_labels)
        input_embeddings = word_embeddings + label_embeddings
        # position_embeddings = self.position_embeddings(x_position_ids)
        # print(tuple_embeddings.shape, user_embeddings.shape)
        if x_user_ids is not None:
            user_embeddings = self.user_embeddings(x_user_ids).unsqueeze(1)
            input_embeddings = input_embeddings + user_embeddings  # + position_embeddings

        hidden_states = None
        if self.encoder_type == 'rnn':
            hidden_states, (h_n, c_n) = self.encoder(input_embeddings)  # states:[batch_size, seq_len, hidden_size]
        elif self.encoder_type == 'transformer':
            attn_mask = torch.triu(torch.ones(input_embeddings.size(1), input_embeddings.size(1)), diagonal=1).bool().to(self.device)
            hidden_states = input_embeddings
            for layer in self.encoder:
                hidden_states, attn_weights = layer(query=hidden_states, key=hidden_states, value=hidden_states, attn_mask=attn_mask)

        logits = self.ffn(hidden_states)  # batch_size, seq_len, num_words
        # update_matrix = torch.tril(torch.ones(input_embeddings.size(1), input_embeddings.size(1)), diagonal=0).to(self.device)  # seq_len, seq_len
        # logits = torch.matmul(update_matrix, logits)    # batch_size, seq_len, num_words

        return logits


class ExerciseGeneratorC(torch.nn.Module):
    def __init__(self, model_name, num_words, use_d, use_a):
        super(ExerciseGeneratorC, self).__init__()
        te_name = 'model.shared.weight'
        pe_name = 'model.encoder.embed_positions.weight'
        self.generator = BartForConditionalGeneration.from_pretrained(model_name)
        self.token_embeddings = self.position_embeddings = None

        for pname, p in self.generator.named_parameters():
            if pname == te_name:
                self.token_embeddings = p
            if pname == pe_name:
                self.position_embeddings = p
        assert self.position_embeddings is not None
        assert self.token_embeddings is not None

        self.dim_input = self.token_embeddings.size(1)
        self.num_words = num_words

        # self.ffn_difficulty = None
        # if use_d:
        #     self.ffn_difficulty = torch.nn.Linear(1, self.dim_input, bias=True)
        self.ffn_difficulty = torch.nn.Linear(1, self.dim_input, bias=True)
        self.ffn_student_state = torch.nn.Linear(num_words, self.dim_input, bias=True)
        # self.ffn_student_state = None
        # if use_a:
        #     self.ffn_student_state = torch.nn.Linear(num_words, self.dim_input, bias=True)

    def forward(self, x_input_ids, x_attention_mask, difficulty, student_state, decoder_input_ids, labels, past_key_values=None, use_cache=False):
        """
        difficulty: [batch_size, ]
        student_state: [batch_size, vocab_size]
        x_input_ids: [batch_size, seq_len]
        x_attention_mask: [batch_size, seq_len]
        decoder_input_ids: [batch_size, seq_len]
        """
        inputs_embeds = self.get_inputs_embeds(x_input_ids, difficulty, student_state)
        outputs = self.generator(
            inputs_embeds=inputs_embeds,
            attention_mask=x_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        # if difficulty is None and student_state is None:
        #     outputs = self.generator(
        #         input_ids=x_input_ids,
        #         attention_mask=x_attention_mask,
        #         decoder_input_ids=decoder_input_ids,
        #         labels=labels,
        #         use_cache=use_cache,
        #         past_key_values=past_key_values
        #     )
        # else:
        #     outputs = self.generator(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=x_attention_mask,
        #         decoder_input_ids=decoder_input_ids,
        #         labels=labels,
        #         use_cache=use_cache,
        #         past_key_values=past_key_values
        #     )
        return outputs

    def get_inputs_embeds(self, x_input_ids, difficulty, student_state):
        # input order: <A, D, S>, position: 1, 2, 3+
        batch_size = x_input_ids.size(0)
        seq_length = x_input_ids.size(1)

        input_embeddings = self.token_embeddings[x_input_ids]

        if student_state is not None and difficulty is not None:
            student_state_vectors = self.ffn_student_state(student_state)
            difficulty_vectors = self.ffn_difficulty(difficulty)
            input_embeddings[:, 1, :] = student_state_vectors
            input_embeddings[:, 2, :] = difficulty_vectors

        elif student_state is not None:
            student_state_vectors = self.ffn_student_state(student_state)
            input_embeddings[:, 1, :] = student_state_vectors
        elif difficulty is not None:
            difficulty_vectors = self.ffn_difficulty(difficulty)
            input_embeddings[:, 1, :] = difficulty_vectors

        input_embeddings += self.position_embeddings[torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)]

        return input_embeddings


class ExerciseGeneratorD(torch.nn.Module):
    def __init__(self, model_name, use_difficulty, max_difficulty_label, d_template):
        super(ExerciseGeneratorD, self).__init__()
        self.generator = BartForConditionalGeneration.from_pretrained(model_name)

        if use_difficulty:
            difficulty_control_tokens = {'additional_special_tokens': [d_template.format(i) for i in range(max_difficulty_label)]}
            self.qg_tokenizer.add_special_tokens(difficulty_control_tokens)
            self.question_generator.resize_token_embeddings(len(self.qg_tokenizer))

    def forward(self, x_input_ids, x_attention_mask, decoder_input_ids, labels):
        outputs = self.generator(
            input_ids=x_input_ids,
            attention_mask=x_attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        return outputs


class Hypothesis:
    def __init__(self, decoded_ids, past_key_values=None, length=0, likelihood=0., difficulty=0.):
        self.decoded_ids = decoded_ids # [decoded_length, ]
        self.likelihood = likelihood
        self.difficulty = difficulty
        self.final_score = 0
        self.past_key_values = past_key_values
        self.length = length
        self.future_soft_difficulty = 0
        self.future_hard_difficulty = 0
        self.future_decoded_ids = 0
        self.final_score = 0

    def compute_score(self, factor, target_difficulty):
        self.final_score = (self.likelihood / len(self.decoded_ids)) / (factor * abs(target_difficulty-self.difficulty))

    def to_str(self, tokenizer):
        return tokenizer.decode(self.decoded_ids)

    def key(self):
        valid_ids = []
        special_ids = [0, 1, 2]
        for sub_word_id in self.decoded_ids.detach().cpu().numpy().tolist():
            if sub_word_id not in special_ids:
                valid_ids.append(str(sub_word_id))
        return '|'.join(valid_ids)

    def __len__(self):
        return self.length


class ConstrainedDecodingWithLookahead:
    def __init__(self, beam_size, word_map, tokenizer, model, factor_d, factor_c, factor_s, lookahead):
        self.beam_size = beam_size  # beam_size
        self.word_map = word_map  # full word
        self.tokenizer = tokenizer  # model's tokenizer
        self.model = model  # generator
        self.special_token_ids = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id, tokenizer.pad_token_id]
        self.factor_c = factor_c  # coverage weight
        self.factor_d = factor_d  # difficulty weight
        self.factor_s = factor_s  # score weight
        self.lookahead_steps = lookahead  # lookahead

    def generate(self, input_ids, attention_mask, target_words, target_difficulty, skill_difficulties, max_steps, use_difficulty, device):
        '''
        input_ids: batch_size, beam_size, seq_len,
        attention_masks: batch_size, beam_size, seq_len
        target_words:
        target_difficulties: [batch_size, 1]
        skill_difficulties: [batch_size, num_words]
        max_steps:
        '''

        # print('search algo Input_ids', input_ids.shape)
        # print('search algo attn_masks', attention_mask.shape)
        # print('search algo target_words', target_words.shape)
        # print('search algo target_difficulty', target_difficulty.shape)
        # print('search algo skill_difficulty', skill_difficulties.shape)

        sub_word_skill_id = self.map_ids(word_map=self.word_map, tokenizer=self.tokenizer, oov_id=self.word_map['<pad>'])  # [len(tokenizer)] sub_words out of skills are set to skill_map[<pad>]
        oov_mask = torch.where(sub_word_skill_id == self.word_map['<pad>'], 0, 1).to(device)
        sub_word_difficulty = skill_difficulties[torch.arange(skill_difficulties.size(0)).unsqueeze(0), sub_word_skill_id] * oov_mask  # [batch_size(1), len(tokenizer)]
        logging.debug('sub_word_difficulty {}'.format(sub_word_difficulty.shape))
        init_beam = Hypothesis(torch.tensor([self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]).to(device))
        beams = [init_beam]

        finished = []
        finished_keys = set()

        for step in range(max_steps):
            logging.debug('{} decoding step {} {}'.format('*'*100, step, '*'*100))
            extensions = []
            for hypo in beams:
                logging.debug('extending hypo: {}'.format(self.tokenizer.decode(hypo.decoded_ids)))
                # last_word_id = hypo.decoded_ids[-1:].unsqueeze(0)  # tensor(cur_len)
                # print('last word id', last_word_id.shape)
                output = self.model(
                    x_input_ids=input_ids,  # [batch_size, seq_len]
                    x_attention_mask=attention_mask,
                    # past_key_values=hypo.past_key_values,
                    # use_cache=True,
                    difficulty=target_difficulty,
                    student_state=None,
                    decoder_input_ids=hypo.decoded_ids.unsqueeze(0),
                    labels=None,
                )
                last_log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)  # [batch_size, 1, vocab_size]
                # print('last_probs', last_probs.shape)
                # values, indices = torch.sort(last_probs, descending=True)
                # print('top 10 values', values[:, :10])
                # print('top 10 indices', indices[:, :10])
                # print('top 10 words', self.tokenizer.batch_decode(indices[:, :10].view(-1, 1)))

                extensions.append((hypo, last_log_probs))

            logging.debug('{} extensions'.format(len(extensions)))
            # prune: preselect top-K across extensions of all beams
            candidate_hypos = self.top_k(extensions, sub_word_difficulties=sub_word_difficulty)  # list of beam_size hypo (for 1 example)
            logging.debug('After prune, {} hypos'.format(len(candidate_hypos)))

            for i, hypo in enumerate(candidate_hypos):
                logging.debug('ids:{}, cur_words:{}, total_length:{}, valid_length:{} difficulty:{}, likelihood: {}'.format(
                    hypo.decoded_ids,
                    self.tokenizer.decode(hypo.decoded_ids),
                    hypo.decoded_ids.size(0),
                    hypo.length,
                    hypo.difficulty,
                    hypo.likelihood,
                ))

            # sort: future difficulty and future constraints (lookahead)
            inputs_embeds = self.model.get_inputs_embeds(
                x_input_ids=input_ids.repeat(len(candidate_hypos), 1),  # [batch_size, input_len]
                difficulty=target_difficulty.unsqueeze(0) if use_difficulty else None,  # [batch_size, 1]
                student_state=None,
            )
            # print('input_embeds', inputs_embeds.shape)
            # print('decoder_input_ids', torch.stack([hypo.decoded_ids for hypo in candidate_hypos], dim=0).shape)
            # exit(1)
            gen_outputs = self.model.generator.generate(
                inputs_embeds=inputs_embeds,
                decoder_input_ids=torch.stack([hypo.decoded_ids for hypo in candidate_hypos], dim=0),
                attention_mask=attention_mask.repeat(len(candidate_hypos), 1),
                num_beams=1,
                max_new_tokens=self.lookahead_steps,
                return_dict_in_generate=True,  # transformers.generation_utils.BeamSearchEncoderDecoderOutput
                output_scores=True,
                num_return_sequences=1
            )  # GreedySearchEncoderDecoderOutput

            # future difficulty
            # sequences = gen_outputs.sequences  # [prune_size, sequence_len]
            # print('scores', len(gen_outputs.scores), gen_outputs.scores[0].shape)  # scores: tuple of [prune_size, vocab_size]
            scores = torch.softmax(torch.stack(gen_outputs.scores, dim=0).permute(1, 0, 2), dim=-1)  # [prune_size, step, vocab_size]
            sequence_difficulties_soft = torch.matmul(scores, sub_word_difficulty.permute(1, 0))  # [prune_size, step, vocab_size] * [vocab_size, 1] = [prune_size, step_size, 1]
            sequence_difficulties_soft = torch.sum(sequence_difficulties_soft, dim=1)
            d_mse = torch.abs(target_difficulty - sequence_difficulties_soft)  # [prune_size, difficulty_mse]

            sequence_difficulties_hard = torch.sum(torch.gather(sub_word_difficulty.repeat(len(candidate_hypos), 1), index=gen_outputs.sequences, dim=-1), dim=-1)

            for i in range(len(candidate_hypos)):
                candidate_hypos[i].future_soft_difficulty = sequence_difficulties_soft[i]
                candidate_hypos[i].future_hard_difficulty = sequence_difficulties_hard[i]
                candidate_hypos[i].future_decoded_ids = gen_outputs.sequences[i]

            # sort candidate
            for hypo in candidate_hypos:
                hypo.final_score = self.get_hypo_score(hypo, target_difficulty, is_finish=False)

            candidate_hypos.sort(
                key=lambda h: h.final_score,
                reverse=True
            )

            beams = candidate_hypos[:self.beam_size]  # TODO: grouping

            # collect finished hypo
            for hypo in beams:
                if hypo.decoded_ids[-1] == self.tokenizer.eos_token_id and hypo.key() not in finished_keys:
                    finished.append(hypo)
                    finished_keys.add(hypo.key())

            logging.debug('{} finished hypos'.format(len(finished)))
            logging.debug('After sorting {} hypos'.format(len(beams)))
            for i, hypo in enumerate(beams):
                logging.debug('ids:{}, cur_words:{}, total_length:{}, valid_length:{} difficulty:{}, likelihood: {}, future_gen {}, future_hard_difficulty: {}, future_soft_difficulty: {}, difficulty_mse: {}'.format(
                    hypo.decoded_ids,
                    self.tokenizer.decode(hypo.decoded_ids),
                    hypo.decoded_ids.size(0),
                    hypo.length,
                    hypo.difficulty,
                    hypo.likelihood,
                    self.tokenizer.decode(hypo.future_decoded_ids),
                    hypo.future_hard_difficulty.detach().cpu().numpy(),
                    hypo.future_soft_difficulty.detach().cpu().numpy(),
                    target_difficulty - hypo.future_soft_difficulty
                ))

            # if step == 1:
            #     break

            if step >= max_steps:
                break

            if len(finished) > self.beam_size:
                break

        for hypo in finished:
            hypo.final_score = self.get_hypo_score(hypo, target_difficulty=target_difficulty, is_finish=True)

        finished.sort(
            key=lambda h: h.final_score,
            reverse=True
        )

        logging.debug('{} finish searching {}'.format('#'*100, '#'*100))
        for i, hypo in enumerate(finished):
            logging.info(
                'key:{}, cur_words:{}, total_length:{}, valid_length:{} difficulty:{}, likelihood: {}, difficulty_mse: {}, final_score {}'.format(
                    hypo.key(),
                    self.tokenizer.decode(hypo.decoded_ids, skip_special_tokens=True),
                    hypo.decoded_ids.size(0),
                    hypo.length,
                    hypo.difficulty,
                    hypo.likelihood,
                    torch.abs(target_difficulty - hypo.difficulty).detach().cpu().numpy(),
                    hypo.final_score
                ))
        exit(1)
        return finished

    def top_k(self, extensions, sub_word_difficulties):
        ext_hypos = []
        # beam_size, vocab_size
        # k-best scores : [batch_size, beam_size*vocab_size] sort_k
        # k-best continuations: [batch_size, beam_size, vocab_size], top_1 along the last dimension
        # k continuations with constraints
        # k continuations with difficulty?

        added_hypos = set()

        # k best continuations
        for hypo, log_probs in extensions:
            values, indices = torch.sort(log_probs, descending=True)
            # print('best continuation', values[0][0], indices[0][0], self.tokenizer.decode(indices[0][0]))
            # print('worst continuation', values[0][-1], indices[0][-1], self.tokenizer.decode(indices[0][-1]))
            # exit(1)
            # print('here', indices[0][0], self.tokenizer.pad_token_id, indices[0][0] == self.tokenizer.pad_token_id)
            new_hypo = Hypothesis(
                torch.cat([hypo.decoded_ids, indices[0][0:1]]),
                length=hypo.length if indices[0][0] in self.tokenizer.all_special_ids else hypo.length+1,
                likelihood=hypo.likelihood if indices[0][0] in self.tokenizer.all_special_ids else hypo.likelihood + values[0][0],
                difficulty=hypo.difficulty + sub_word_difficulties[0, indices[0][0]]  # TODO: batch
            )
            ext_hypos.append(new_hypo)
            added_hypos.add(new_hypo.key())

        # k-best likelihoods
        full_likelihoods = torch.cat([hypo.likelihood + log_probs for hypo, log_probs in extensions], dim=-1)  # [beam_size * vocab_size]
        values, indices = torch.sort(full_likelihoods, descending=True)  # values: sum of log
        for i in range(self.beam_size):
            hypo = extensions[torch.div(indices[0][i], len(self.tokenizer), rounding_mode='trunc')][0]
            sub_word_id = indices[0][i] % len(self.tokenizer)
            new_hypo = Hypothesis(
                decoded_ids=torch.cat([hypo.decoded_ids, sub_word_id.unsqueeze(0)]),
                length=hypo.length if sub_word_id in self.tokenizer.all_special_ids else hypo.length + 1,
                likelihood=hypo.likelihood if sub_word_id in self.tokenizer.all_special_ids else hypo.likelihood + values[0][i],
                difficulty=hypo.difficulty + sub_word_difficulties[0][sub_word_id]  # TODO: batch
            )
            if new_hypo.key() not in added_hypos:
                ext_hypos.append(new_hypo)
                added_hypos.add(new_hypo.key())

        return ext_hypos

    def get_hypo_score(self, hypo, target_difficulty, is_finish):
        avg_log_likelihood = hypo.likelihood / hypo.length
        difficulty_consistency = target_difficulty - hypo.difficulty if is_finish else hypo.difficulty + hypo.future_soft_difficulty
        return - self.factor_d * torch.abs(difficulty_consistency) + self.factor_s * avg_log_likelihood

    @classmethod
    def map_vocab(cls, word_map, student_state, tokenizer, unk_word_difficulty, special_word_difficulty=0):
        # map known word difficulty to tokenizer vocabulary
        # sub-word difficulty == word difficulty
        sub_word_difficulty_map = [unk_word_difficulty for i in range(len(tokenizer))]

        for special_token_id in tokenizer.all_special_ids:
            sub_word_difficulty_map[special_token_id] = special_word_difficulty

        inverse_word_map = {word_map[word]: word for word in word_map}
        for word_id in range(len(student_state)):  # skill_id
            word = inverse_word_map[word_id]
            formats = [word, ' ' + word, word.capitalize(), ' ' + word.capitalize()]
            for f in formats:
                sub_word_ids = tokenizer(f)['input_ids'][1:-1]
                for sub_word_id in sub_word_ids:
                    sub_word_difficulty_map[sub_word_id] = student_state[word_id]

        return torch.tensor(sub_word_difficulty_map)

    @classmethod
    def map_ids(cls, word_map, tokenizer, oov_id):
        # map known word difficulty to tokenizer vocabulary
        # sub-word difficulty == word difficulty
        indices = [oov_id for i in range(len(tokenizer))]
        inverse_word_map = {word_map[word]: word for word in word_map}
        word_type_record = {}
        for word_id in range(len(word_map)):  # skill_id
            word = inverse_word_map[word_id]
            formats = [word, ' ' + word, word.capitalize(), ' ' + word.capitalize()]
            for f in formats:
                sub_word_ids = tokenizer(f)['input_ids'][1:-1]
                for sub_word_id in sub_word_ids:
                    if indices[sub_word_id] != oov_id:
                        if word_type_record[sub_word_id] == 'whole':
                            continue  # whole word > sub word
                    indices[sub_word_id] = word_id
                    if len(sub_word_ids) == 1:
                        word_type_record[sub_word_id] = 'whole'
                    else:
                        word_type_record[sub_word_id] = 'sub'

        return torch.tensor(indices)