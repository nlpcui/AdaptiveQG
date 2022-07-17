import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    # Transformer PE
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)


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
    def __init__(self, num_emb, dim_emb, max_len, dim_pe, num_exercise_encoder_layers, dim_attn_exercise, num_attn_heads_exercise, dim_ff_exercise, dropout_exercise, num_interaction_encoder_layers, dim_attn_interaction, num_attn_heads_interaction, dim_ff_interaction, dropout_interaction, num_label):
        super(SAKT, self).__init__()
        self.embeddings = nn.Embedding(num_emb, dim_emb) # |formats|+|students|+|words|+|interactions|+|pad|+|unk|
        self.positional_embedding = nn.Embedding(max_len, dim_pe)
        # self.temporal_embedding = nn.Embedding(num_time, dim_te)
        # self.memory_key = nn.Embedding(num_words)
        
        self.exercise_encoder = nn.ModuleList([
            TransformerEncodingBlock(dim_attn_exercise, num_attn_heads_exercise, dim_ff_exercise, dropout_exercise) for i in range(num_exercise_encoder_layers)
        ]) # encode <word, label> sequence
        self.interaction_encoder = nn.ModuleList([
            TransformerEncodingBlock(dim_attn_interaction, num_attn_heads_interaction, dim_ff_interaction, dropout_interaction) for i in range(num_interaction_encoder_layers)
        ]) # encode <word> sequence

        self.ff_output = nn.Linear(dim_attn_interaction, num_label, bias=True) 

    
    def forward(self, x_exercise, x_interaction, x_exercise_attn_mask, x_interaction_attn_mask, y_labels):
        # x_exercise: [batch_size, max_seq_len]
        # x_interaction: [batch_size, max_seq_len]
        # x_exercise_attn_mask: [batch_size, max_seq_len, max_seq_len]
        # x_interaction_attn_mask: [batch_size, max_seq_len, max_seq_len]
        # y_labels: [batch_size, seq_len]

        exercise_emb_seq = self.embeddings(x_exercise)
        interaction_emb_seq = self.embeddings(x_interaction)
        
        # positional_embs = self.positional_embedding(x_token)
        # temporal_embs = self.temporal_embedding(x_token)
        # token_emb = token_emb + temporal_emb + positional_emb
        # token_label_emb = token_label_emb + temporal_emb + positional_emb

        # encoding exercise context
        encoded_exercise = exercise_emb_seq
        for layer_id, exercise_encoder_layer in enumerate(self.exercise_encoder):
            # print(layer_id, encoded_exercise.shape)
            encoded_exercise, attn_weight_exercise = exercise_encoder_layer(
                query=encoded_exercise,
                key=encoded_exercise,
                value=encoded_exercise,
                attn_mask=x_exercise_attn_mask # only words in the same exercise are attended
            )
        # encoding history interaction
        for layer_id, interaction_encoder_layer in enumerate(self.interaction_encoder):
            # print(layer_id, encoded_exercise.shape)
            encoded_exercise, attn_weight_exercise = interaction_encoder_layer(
                query=encoded_exercise,
                key=interaction_emb_seq,
                value=interaction_emb_seq,
                attn_mask=x_interaction_attn_mask # only words in the same exercise are attended
            )
        
        logits = self.ff_output(encoded_exercise) # [batch_size, seq_len, num_label] 

        return logits


class DKT(nn.Module):
    def __init__(self):
        super(SAKT, self).__init__()
        pass

    
    def forward(self):
        pass



class DKT(nn.Module):
    def __init__(self):
        super(DKT, self).__init__()



# sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_, index):
        a = torch.Linear(input_.size(-1), 1)
        a = self.sigmoid(a)
        logits = torch.masked_select(a, index)
        return logits
        

# print(attn_mask.shape)
# layer = nn.MultiheadAttention(5, 1, batch_first=True)
# attn_outputs, attn_weights = layer(query, query, query, attn_mask=attn_mask)
# print(attn_weights)
# print(attn_weights.shape)
# N = 10
# criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
# groundtruth = torch.rand(N, ).ge(0.5).type(torch.LongTensor)
# groundtruth[7:] = -1
# pred = torch.rand(N, 2)
# print(groundtruth, pred)
# loss = criterion(pred, groundtruth)
# print(loss)

# index = torch.tensor([[0, 1], [2, 1]])
# a = torch.tensor([[[0.2, 0.8, 0.4], [0.4, 0.6, 0.5]], [[0.1, 0.9, 0.2], [0.3, 0.7, 0.4]]], requires_grad=True) # batch_size, 
# # print(a.requires_grad)
# # exit(1)
# select = torch.tensor([[[1, 1, 1], [0, 0, 0]], [[1, 1, 1], [0, 0, 0]]]) > 0

# max_ = torch.argmax(a)
# print(max_.requires_grad)

# net = Net()
# gradecheck(net, (a, index))

# d0 = torch.arange(a.size(0))
# out = a[[0, 1], [[0, 1], [1, 1]], :]

# print(out.requires_grad)

# print(a.shape)



# selected = torch.masked_select(a, select)

# print(selected.requires_grad)