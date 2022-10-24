import torch, math, transformers, logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig, BartTokenizer
import numpy as np
import pandas as pd



class CrossEntropyFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='mean', ignore_index=-100):
        super(CrossEntropyFocalLoss, self).__init__()
        self.reduction = reduction 
        self.lb_ignore = ignore_index
        self.gamma = gamma # focal factor
        self.alpha = alpha # class weight, tensor
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0

            lb_pos, lb_neg = 1. , 0.
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)

        if self.alpha is not None:
            alpha = self.alpha.gather(0, label.detach().view(-1))
            n_valid = alpha.sum()
            logs = logs * alpha

        pt = logs.detach().exp()
        focal_factor = (1-pt) ** self.gamma

        loss = -torch.sum(focal_factor * logs * lb_one_hot, dim=1)

        loss[ignore] = 0 # ignore_index no loss
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss
    

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


class SAKT(nn.Module):
    def __init__(self, num_words, num_w_l_tuples, num_tasks, num_time, num_days, num_users, max_seq_len, dim_emb_word, dim_emb_tuple, dim_emb_task, dim_emb_position, dim_emb_time, dim_emb_days, dim_emb_user, num_exercise_encoder_layers, dim_attn_exercise, num_attn_heads_exercise, dim_ff_exercise, dropout_exercise, num_interaction_encoder_layers, dim_attn_interaction, num_attn_heads_interaction, dim_ff_interaction, dropout_interaction, ftr_comb='sum', device='cpu', num_labels=2, alpha=0.8, emb_padding_idx=0):
        super(SAKT, self).__init__()
        
        self.alpha = alpha
        self.device = device
        self.ftr_comb = ftr_comb

        self.word_embeddings = nn.Embedding(num_words, dim_emb_word, padding_idx=emb_padding_idx)
        self.w_l_tuple_embeddings = nn.Embedding(num_w_l_tuples, dim_emb_tuple, padding_idx=emb_padding_idx) 
        self.time_embeddings = nn.Embedding(num_time, dim_emb_time, padding_idx=emb_padding_idx)
        self.days_embeddings = nn.Embedding(num_days, dim_emb_days, padding_idx=emb_padding_idx)
        self.user_embeddings = nn.Embedding(num_users, dim_emb_user)
        self.positional_embeddings = nn.Embedding(max_seq_len, dim_emb_position, device=device)
        # self.positional_embeddings = PostionalEncoding(dim_emb, max_seq_len=max_seq_len, device=device)
        self.task_embeddings = nn.Embedding(num_tasks, dim_emb_task, padding_idx=emb_padding_idx)
        
        self.word_encoder = nn.ModuleList([
            TransformerEncodingBlock(dim_attn_exercise, num_attn_heads_exercise, dim_ff_exercise, dropout_exercise) for i in range(num_exercise_encoder_layers)
        ]) # self-attention in words
        self.w_l_tuple_encoder = nn.ModuleList([
            TransformerEncodingBlock(dim_attn_interaction, num_attn_heads_interaction, dim_ff_interaction, dropout_interaction) for i in range(num_interaction_encoder_layers)
        ]) # self-attention in (word, label) tuples

        self.cross_encoder = nn.ModuleList([
            TransformerEncodingBlock(dim_attn_exercise, num_attn_heads_exercise, dim_ff_exercise, dropout_exercise) for i in range(num_exercise_encoder_layers)
        ]) # cross-attention (query=word, key,value=tuples)

        self.ff_output_context_1 = nn.Linear(dim_attn_interaction, 2*dim_attn_interaction, bias=True)
        self.ff_output_context_2 = nn.Linear(2*dim_attn_interaction, num_labels, bias=True)
        
        self.ff_output_memory_pos_1 = nn.Linear(dim_attn_interaction, 2*num_words, bias=True)
        self.ff_output_memory_pos_2 = nn.Linear(2*num_words, num_words, bias=True)

        self.ff_output_memory_neg_1 = nn.Linear(dim_attn_interaction, 2*num_words, bias=True)
        self.ff_output_memory_neg_2 = nn.Linear(2*num_words, num_words, bias=True)


    def forward(self, x_word_ids, x_word_attn_masks, x_w_l_tuple_ids, x_w_l_tuple_attn_masks, x_position_ids, x_task_ids, x_user_ids, x_days, x_time, x_interaction_ids, x_memory_update_matrix):
        '''
        x_word_ids:                 [batch_size, seq_len]
        x_w_l_tuple_ids:            [batch_size, seq_len]
        x_position_ids:             [batch_size, seq_len]
        x_task_ids:                 [batch_size, seq_len]
        x_exercise_attn_mask:       [batch_size*attn_heads, seq_len, seq_len]
        x_interaction_attn_mask:    [batch_size*attn_heads, seq_len, seq_len]
        x_interaction_ids:          [batch_size, seq_len]
        x_sep_indices:              [batch_size*attn_heads, num_interactions]
        '''

        batch_size = x_word_ids.size(0)
        seq_len = x_word_ids.size(1)

        word_embs = self.word_embeddings(x_word_ids) 
        w_l_tuple_embs = self.w_l_tuple_embeddings(x_w_l_tuple_ids) 
        
        pos_embs = self.positional_embeddings(x_position_ids)
        task_embs = self.task_embeddings(x_task_ids)
        days_embs = self.days_embeddings(x_days)
        time_embs = self.time_embeddings(x_time)
        user_embs = self.user_embeddings(x_user_ids).unsqueeze(1)        

        if self.ftr_comb == 'sum':
            # print('w_l_tuple_embs', w_l_tuple_embs.shape)
            # print('task_embs', task_embs.shape)
            # print('pos_embs', pos_embs.shape)
            # print('days_embs', days_embs.shape)
            # print('time_embs', time_embs.shape)
            # print('user_embs', user_embs.shape)
            w_l_tuple_embs = w_l_tuple_embs + task_embs + pos_embs + days_embs + time_embs + user_embs
            word_embs = word_embs + task_embs + pos_embs + days_embs + time_embs + user_embs

        elif self.ftr_comb == 'concat':
            pass
        
        # intra-exercise context transformer
        # h_word_state = word_embs
        # for layer_id, word_encoding_layer in enumerate(self.word_encoder):
        #     h_word_state, cross_attn_weights = word_encoding_layer(query=h_word_state, key=h_word_state, value=h_word_state, attn_mask=x_word_attn_masks)
        
        # print('h_word_state has nan? {}'.format(torch.isnan(h_word_state).any()))
        # causal interaction transformer
        # h_w_l_tuple_state = w_l_tuple_embs
        # for layer_id, w_l_tuple_encoding_layer in enumerate(self.w_l_tuple_encoder):
        #     h_w_l_tuple_state, cross_attn_weights = w_l_tuple_encoding_layer(query=h_w_l_tuple_state, key=h_w_l_tuple_state, value=h_w_l_tuple_state, attn_mask=x_w_l_tuple_attn_masks)
        # print('h_tuple_state has nan? {}'.format(torch.isnan(h_w_l_tuple_state).any()))
        
        # cross causal <word, interaction> transformer
        h_cross_state = word_embs
        for layer_id, cross_encoding_layer in enumerate(self.cross_encoder):
            h_cross_state, cross_attn_weights = cross_encoding_layer(query=h_cross_state, key=w_l_tuple_embs, value=w_l_tuple_embs, attn_mask=x_w_l_tuple_attn_masks)
        # print('h_cross_state has nan? {}'.format(torch.isnan(h_cross_state).any()))

        logits_context = self.ff_output_context_2(torch.tanh(self.ff_output_context_1(h_cross_state))) # [batch_size, max_seq_len, 2]
        # print('logits_context has nan? {}'.format(torch.isnan(logits_context).any()))
        memory_states_pos = self.ff_output_memory_pos_2(torch.tanh(self.ff_output_memory_pos_1(h_cross_state))) # .unsqueeze(-1) # [batch_size, seq_len, num_words]
        memory_states_pos_acc = torch.matmul(memory_states_pos.permute(0, 2, 1), x_memory_update_matrix).permute(0, 2, 1)

        memory_states_neg = self.ff_output_memory_neg_2(torch.tanh(self.ff_output_memory_neg_1(h_cross_state))) #.unsqueeze(-1) # [batch_size, seq_len, num_words]
        memory_states_neg_acc = torch.matmul(memory_states_neg.permute(0, 2, 1), x_memory_update_matrix).permute(0, 2, 1) # [batch_size, seq_len, num_words]

        memory_states_acc = torch.stack([memory_states_pos_acc, memory_states_neg_acc], dim=-1)

        index = x_word_ids.unsqueeze(-1)
        word_mastery_pos_acc = torch.gather(memory_states_pos_acc, dim=-1, index=index) # [batch_size, seq_len, 1]
        word_mastery_neg_acc = torch.gather(memory_states_neg_acc, dim=-1, index=index) # [batch_size, seq_len, 1]
        logits_memory = torch.cat([word_mastery_pos_acc, word_mastery_neg_acc], dim=-1) # [batch_size, seq_len, 2]

        # print('logits_memory has nan? {}'.format(torch.isnan(logits_memory).any()))
        logits = self.alpha * logits_memory + (1 - self.alpha) * logits_context # [batch_size, max_seq_len, 2]

        # print('logits has nan? {}'.format(torch.isnan(logits).any()))
        # exit(1)
        return logits, memory_states_acc


class DKT(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, num_tuples, output_size, ceil='LSTM'):
        super(DKT, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=True)
        self.tuple_embeddings = nn.Embedding(num_tuples, input_size)
        self.ffn = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x_w_l_tuple_ids):
        tuple_embeddings = self.tuple_embeddings(x_w_l_tuple_ids)
        states, (h_n, c_n) = self.rnn(tuple_embeddings) # states:[batch_size, seq_len, hidden_size]
        logits = self.ffn(states)

        return logits


class ReWeightBlock(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        self.ff1 = nn.Linear(dim_input, dim_hidden, bias=True)
        self.ff2 = nn.Linear(dim_hidden, dim_output, bias=True)
        
    def forward(self, x):
        return x + self.ff2(torch.relu(self.ff1(x))) 


class AdaptiveQuestionGenerator(nn.Module):
    def __init__(self, model_name, vocab_size, temperature):
        super(QuestionGenerator, self).__init__()
        self.base_generator = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
        self.temperature = temperature
        self.vocab_size = vocab_size
        self.reweighter = torch.nn.ModuleList([
            ReWeightBlock(vocab_size*3, vocab_size*3, vocab_size)
        ])
        

    def forward(self, x_keyword_ids, x_attention_mask, x_knowledge_state, y_difficulties, y_exercise_ids, decoder_input_ids):
        '''
        x_keyword_ids: [batch_size, x_max_length]
        x_attention_mask: [batch_size, x_max_length]
        x_knowlegde_states: [batch_size, vocab_size]
        x_difficulties: [batch_size, 1]
        y_exercises: [batch_size, y_max_length]
        decoder_input_ids: [batch_size, y_max_length]
        
        '''
        outputs = self.base_generator(
            input_ids=x_keyword_ids, 
            attention_mask=x_attention_mask, 
            decoder_input_ids=decoder_input_ids,
            labels=y_exercise_ids
        ) 
        # batch_size, seq_len, vocab_size
        previous_generations = torch.softmax(outputs.logits/self.temperature)

        outputs.logits
        return outputs


class Hypothesis:
    def __init__(self, decoded_ids, likelihood=0, difficulty=0):
        self.decoded_ids = decoded_ids
        self.likelihood = likelihood
        self.difficulty = difficulty
        self.final_score = 0

    def extends(self, probs, difficulties):
        extensions = []
        for i in range(len(probs)):
            extensions.append(Hypothesis(
                decoded_ids=self.decoded_ids + [i],
                likelihood=self.likelihood + math.log(probs[i]),
                difficulty=self.difficulty + difficulties[i]
            ))
        
        return extensions

    def compute_score(self, factor, target_value):
        self.final_score = self.likelihood / len(self.decoded_ids) + factor * abs(target_value-self.difficulty)


class SubWordItem:
    def __init__(self, subword_id, subword_match, subword_type, subword_value):
        self.subword_id = subword_id
        self.subword_match = subword_match # list
        self.subword_type = subword_type # n: not in vocab; b: subword_start; i: subword_inside, o: complete word  
        self.subword_value = subword_value # 0 for special_tokens, -99999 for oo_source 


class BeamSearchWithDynamicGrouping:
    def __init__(self, model, beam_size, temperature, score_bucket, source_vocab, tokenizer, max_steps):
        self.beam_size = beam_size
        self.temperature = temperature
        self.vocab_size = len(tokenizer)
        self.score_bucket = score_bucket
        self.tokenizer = tokenizer
        self.model = model
        self.special_token_ids = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id, tokenizer.pad_token_id]
        self.max_steps = max_steps

        self.source_vocab = source_vocab
        self.target_vocab = {i: SubWordItem(i, [], None, -99999) for i in range(len(self.tokenizer))}
        
        self.map_vocab()


    def generate(self, input_ids, attention_mask, probs, target_value, vocab_value, num_return):
        '''
        logits: seq_len, vocab_size
        '''
        # set value of self.target_vocab
        self.__set_value(vocab_value)

        num_groups = int(target_value//bucket+1) 
        hypo_groups = [[Hypothesis(tokenizer.bos_token_id)]]
        
        qualified_hypotheses = []

        past_key_values = None
        is_finish = False
        step = 0

        while not is_finish:
            extensions = []
            for group in hypo_groups:
                for hypo in group:
                    last_word_id = hpyo.decoded_ids[:, [-1], :]
                    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values, decoder_input_ids=last_word_id)
                    last_logits = output.logits[:, [-1], :] # [batch_size, 1, vocab_size]
                    last_probs = torch.softmax(last_logits / self.temperature)
                    extensions.extend(hypo.extends(last_probs, vocab_score))

            hypo_groups = self.regroup(extensions)
            group_size = self.allocate_group_size(hypo_groups)
            self.ingroup_rank(hypo_groups, group_size)
            
            step += 1
            # finish criteria
            if step >= self.max_steps:
                is_finish = True
            elif has_qualified_results(result_collections):
                is_finish = True

        # final selection
        for hypo in qualified_hypotheses:
            hypo.compute_score(self.match_factor, target_value)
        qualified_hypotheses.sort(key=lambda x:x.final_score)
        
        return qualified_hypotheses[: num_return]


    def regroup(self, num_groups):
        # regroup
        # 1.regroup extensions to groups; 2.adjust group size; 3.in-group sorting, 
        extensions = []
        groups = [[] i for i in range(num_groups)]
        for hypo in extensions:
            pass


    def allocate_group_size(self, hypo_groups):
        pass
        return group_size


    def ingroup_rank(self, hypo_groups, group_size):
        pass
        return completions


    def map_vocab(self):
        # map source vocab (skills) to target vocab (tokenizer vocab)
        '''
        special_tokens, map=[]
        oov_tokens, map=[]
        match_tokens, map=[skill_ids]
        '''
        
        match_words = {}
        unmatch_words = {word: 0 for word in self.source_vocab}
        # print(unmatch_words)

        # complete word match
        for subword_id in range(len(self.tokenizer)):
            if subword_id in self.special_token_ids:
                continue # special_tokens, map=[]
            else:
                subword = self.tokenizer.decode(subword_id).strip().lower()
                if subword in self.source_vocab: # matched complete word
                    self.target_vocab[subword_id].subword_match.append(subword)
                    self.target_vocab[subword_id].subword_type = 'n'
                    match_words[subword] = self.source_vocab[subword]
                    if subword in unmatch_words:
                        unmatch_words.pop(subword)
                else: # not matched word
                    pass # TODO: handle subword

        # subword match
        for original_word in unmatch_words:
            word_formats = [original_word, original_word.capitalize(), ' '+original_word, ' '+original_word.capitalize()]
            for word in word_formats:
                subword_ids = self.tokenizer(word, add_special_tokens=False)['input_ids']
                for subword_id in subword_ids:
                    # if self.tokenizer.decode(subword_id) in match_words:
                    #     print('subword is also full word', self.tokenizer.decode(subword_id))
                    if word.startswith(' ') or word[0].isupper():
                        self.target_vocab[subword_id].subword_type = 'b'
                    else:
                        self.target_vocab[subword_id].subword_type = 'i'
                    if original_word not in self.target_vocab[subword_id].subword_match:
                        self.target_vocab[subword_id].subword_match.append(original_word)


        # for word_id in self.target_vocab:
        #     if len(self.target_vocab[word_id]) > 1:
        #         print('"{}"'.format(word_id), '"{}"'.format(self.tokenizer.decode(word_id)), self.target_vocab[word_id])
        
        # exit(1)
        # cnt = 0
        # mcnt = 0
        # for word in self.target_vocab:
        #     if len(self.target_vocab[word]) > 0:
        #         cnt += 1
        #     if len(self.target_vocab[word]) > 1:
        #         mcnt += 1
        # print(cnt, mcnt)
        # exit(1)
        return match_words, unmatch_words

    def assign_values(self, source_vocab_value):
        for word_id in self.target_vocab:
            avg_value = 0
            for match in self.target_vocab[word_id].subword_match:
                avg_value += source_vocab_value[match]
            avg_value /= len(self.target_vocab[word_id].subword_match)

            self.target_vocab[word_id].value = avg_value

'''
baselines
'''


if __name__ == '__main__':
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    df = pd.read_csv('/Users/cuipeng/Documents/Datasets/duolingo_2018_shared_task/data_en_es/words.csv')
    vocabulary = {}
    for _, row in df.iterrows():
        vocabulary[row['word']] = row['error_rate']
        
    generator = BeamSearchWithDynamicGrouping(model=None, beam_size=5, temperature=1, score_bucket=0.2, source_vocab=vocabulary,  tokenizer=bart_tokenizer, max_steps=50)