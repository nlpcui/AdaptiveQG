import torch, math, transformers, logging
import torch.nn as nn
import torch.nn.functional as F
# from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig, BartTokenizer
import numpy as np


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


class KnowledgeTracer(nn.Module):
    def __init__(self, num_words, num_w_l_tuples, num_tasks, max_seq_len, dim_emb, num_exercise_encoder_layers, dim_attn_exercise, num_attn_heads_exercise, dim_ff_exercise, dropout_exercise, num_interaction_encoder_layers, dim_attn_interaction, num_attn_heads_interaction, dim_ff_interaction, dropout_interaction, device='cpu', num_labels=2, alpha=0.8, emb_padding_idx=0):
        super(KnowledgeTracer, self).__init__()
        
        self.alpha = alpha
        self.device = device

        self.word_embeddings = nn.Embedding(num_words, dim_emb, padding_idx=emb_padding_idx)
        self.w_l_tuple_embeddings = nn.Embedding(num_w_l_tuples, dim_emb, padding_idx=emb_padding_idx) 

        self.positional_embeddings = PostionalEncoding(dim_emb, max_seq_len=max_seq_len, device=device)
        self.task_embeddings = nn.Embedding(num_tasks, dim_emb, padding_idx=emb_padding_idx)
        
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


    def forward(self, x_word_ids, x_word_attn_masks, x_w_l_tuple_ids, x_w_l_tuple_attn_masks, x_position_ids, x_task_ids, x_interaction_ids, x_memory_update_matrix):
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

        pos_embs = self.positional_embeddings(x_word_ids)
        task_embs = self.task_embeddings(x_task_ids)

        word_embs = self.word_embeddings(x_word_ids) 
        w_l_tuple_embs = self.w_l_tuple_embeddings(x_w_l_tuple_ids) 
        

        w_l_tuple_embs = w_l_tuple_embs + task_embs + pos_embs
        word_embs = word_embs + task_embs + pos_embs

        # intra-exercise context transformer
        h_word_state = word_embs
        for layer_id, word_encoding_layer in enumerate(self.word_encoder):
            h_word_state, cross_attn_weights = word_encoding_layer(query=h_word_state, key=h_word_state, value=h_word_state, attn_mask=x_word_attn_masks)
        
        # causal interaction transformer
        h_w_l_tuple_state = w_l_tuple_embs
        for layer_id, w_l_tuple_encoding_layer in enumerate(self.w_l_tuple_encoder):
            h_w_l_tuple_state, cross_attn_weights = w_l_tuple_encoding_layer(query=h_w_l_tuple_state, key=h_w_l_tuple_state, value=h_w_l_tuple_state, attn_mask=x_w_l_tuple_attn_masks)
        
        # cross causal <word, interaction> transformer
        h_cross_state = h_word_state
        for layer_id, cross_encoding_layer in enumerate(self.cross_encoder):
            h_cross_state, cross_attn_weights = cross_encoding_layer(query=h_cross_state, key=h_w_l_tuple_state, value=h_w_l_tuple_state, attn_mask=x_w_l_tuple_attn_masks)

        logits_context = self.ff_output_context_2(nn.functional.tanh(self.ff_output_context_1(h_cross_state))) # [batch_size, max_seq_len, 2]

        memory_states_pos = self.ff_output_memory_pos_2(nn.functional.tanh(self.ff_output_memory_pos_1(h_cross_state))).unsqueeze(-1) # [batch_size, seq_len, num_words, 1]
        memory_states_neg = self.ff_output_memory_neg_2(nn.functional.tanh(self.ff_output_memory_neg_1(h_cross_state))).unsqueeze(-1) # [batch_size, seq_len, num_words, 1]
        memory_states = torch.cat([memory_states_pos, memory_states_neg], dim=-1) # [batch_size, num_interactions, num_words, 2]

        # print('state', memory_states.shape, memory_states.dtype)
        # print('update_matrix', x_memory_update_matrix.shape, x_memory_update_matrix.dtype)
        memory_states = torch.matmul(memory_states.permute(0, 3, 2, 1), x_memory_update_matrix).permute(0, 3, 2, 1) # [batch_size, num_interactions, num_words, 2]
        # print('state2', memory_states.shape, memory_states.dtype)
        '''
        # memory_states = h_cross_state[torch.arange(0, batch_size).unsqueeze(-1), x_sep_indices] # [batch_size, num_interactions, hidden_size]
        # memory_states_pos = self.ff_output_memory_pos_2(nn.functional.tanh(self.ff_output_memory_pos_1(memory_states))).unsqueeze(-1) # [batch_size, num_interactions, num_words, 1]
        # memory_states_neg = self.ff_output_memory_neg_2(nn.functional.tanh(self.ff_output_memory_neg_1(memory_states))).unsqueeze(-1) # [batch_size, num_interactions, num_words, 1]
        # memory_states = torch.cat([memory_states_pos, memory_states_neg], dim=-1) # [batch_size, num_interactions, num_words, 2]
        # x_memory_update_triu = torch.triu(torch.ones(x_sep_indices.size(1), x_sep_indices.size(1)), diagonal=0).to(self.device) # [num_interactions, num_interactions]
        # memory_states = torch.matmul(memory_states.permute(0, 3, 2, 1), x_memory_update_triu).permute(0, 3, 2, 1) # [batch_size, num_interactions, num_words, 2]
        '''
        
        logits_memory = memory_states[torch.arange(0, batch_size), torch.arange(0, seq_len), x_word_ids] # [batch_size, max_seq_len, 2] 

        # print('lm', logits_memory.shape)
        # print('lc', logits_context.shape)

        # exit(1)
        logits = self.alpha * logits_memory + (1 - self.alpha) * logits_context # [batch_size, max_seq_len, 2]

        return logits, memory_states



class AdaptiveQuestionGenerator(nn.Module):
    def __init__(self, model_name):
        super(QuestionGenerator, self).__init__()
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(model_name, output_attentions=True)
        

    def forward(self, x_keyword_ids, x_attention_mask, x_knowledge_state, y_difficulties, y_exercise_ids, decoder_input_ids):
        '''
        x_keyword_ids: [batch_size, x_max_length]
        x_attention_mask: [batch_size, x_max_length]
        x_knowlegde_states: [batch_size, vocab_size]
        x_difficulties: [batch_size, 1]
        y_exercises: [batch_size, y_max_length]
        decoder_input_ids: [batch_size, y_max_length]
        
        '''
        outputs = self.generator(
            input_ids=x_keyword_ids, 
            attention_mask=x_attention_mask, 
            decoder_input_ids=decoder_input_ids,
            labels=y_exercise_ids
        )

        return outputs


class AQG(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


'''
baselines
'''


class NonAdaptiveQuestionGenerator(nn.Module):
    def __init__(self, model_name, num_difficulty_levels, num_embeddings, enable_difficulty):
        # model options: T5, Bart, 
        config = AutoConfig.from_pretrained(model_name)
        self.generator = AutoModelForSeq2SeqLM.from_config(model_name)


    def forward(self):
        
        pass


## Bart


## T5


## GPT2
