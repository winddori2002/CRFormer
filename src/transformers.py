import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class FeatureEmbedding(nn.Module):
    def __init__(self, f_sizes, d_model, dropout):
        super().__init__()
        
        self.feat_num = len(f_sizes)
        self.feat_emb = nn.ModuleList([nn.Embedding(i, d_model) for i in f_sizes])
        self.linear   = nn.Sequential(nn.Linear(d_model*self.feat_num, d_model), nn.BatchNorm1d(d_model), nn.ReLU())

    def forward(self, features):
        
        outputs = []
        for i in range(self.feat_num):
            outputs.append(self.feat_emb[i](features[:,i]))
        outputs = torch.hstack(outputs)
        outputs = self.linear(outputs)

        return outputs
    
class NestedSequenceEmbedding(nn.Module):
    def __init__(self, input_size, d_model, dropout=0.1):
        super(NestedSequenceEmbedding, self).__init__()
        
        self.dropout0 = nn.Dropout(dropout)
        self.rnn_emb0 = getattr(nn, 'GRU')(input_size=60, hidden_size=d_model, batch_first=True, num_layers=1, dropout=dropout, bidirectional=False)  
        self.dropout1 = nn.Dropout(dropout)
        self.rnn_emb1 = getattr(nn, 'GRU')(input_size=input_size, hidden_size=d_model, batch_first=True, num_layers=1, dropout=dropout, bidirectional=False)       
        self.oca      = MultiHeadAttention(d_model, 1, d_model, dropout) # One-sided Cross Attention (OCA)

    def forward(self, x):
        
        b,l1,l2   = x.size()
        
        x_main, _ = self.rnn_emb0(self.dropout0(x))        
        
        x_sub     = self.dropout1(x).view(b*l1,l2,1)
        x_sub, _  = self.rnn_emb1(x_sub)
        x_sub     = x_sub.view(b,l1,l2,-1)
        x_sub     = x_sub.mean(dim=2) + x_sub.max(dim=2)[0]
        
        x_main, _ = self.oca(x_main, x_sub, x_sub)
        
        return x_main
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout     = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn   = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k    = d_k

        self.q_linear   = nn.Linear(d_model, n_head*d_k, bias=False)
        self.k_linear   = nn.Linear(d_model, n_head*d_k, bias=False)
        self.v_linear   = nn.Linear(d_model, n_head*d_k, bias=False)
        self.attn       = ScaledDotProductAttention(temperature=d_k**0.5)
        self.linear     = nn.Linear(n_head*d_k, d_model, bias=False)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        
        batch_size = q.size(0) 
        residual   = q

        q = self.q_linear(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attn(q, k, v, mask=mask)
        
        q = q.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_k)
        q = self.dropout(self.linear(q))
        q = q + residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseConvLayer(nn.Module):
    def __init__(self, d_model, d_ffn, activation, dropout=0.1):
        super(PositionwiseConvLayer, self).__init__()
        
        self.conv1      = nn.Conv1d(in_channels=d_model, out_channels=d_ffn, kernel_size=1)
        self.conv2      = nn.Conv1d(in_channels=d_ffn, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):

        residual = x
        
        x = self.dropout(self.activation(self.conv1(x.transpose(-1,-2))))
        x = self.dropout(self.conv2(x).transpose(-1,-2))
        x = x + residual
        x = self.layer_norm(x)

        return x
    
# Belows for other benchmarks
    
class SequenceEmbedding(nn.Module):
    def __init__(self, input_size, d_model, dropout=0.1):
        super(SequenceEmbedding, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.rnn_emb = getattr(nn, 'GRU')(input_size=input_size, hidden_size=d_model, batch_first=True, num_layers=1, dropout=dropout, bidirectional=True)
        self.linear  = nn.Sequential(nn.Linear(d_model*2, d_model), nn.LayerNorm(d_model), nn.ReLU())

    def forward(self, x):
        x, _ = self.rnn_emb(self.dropout(x))
        x    = self.linear(x)
        return x
    
class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_emb = ValueEmbedding(c_in, d_model)
        self.pos_emb   = PositionalEmbedding(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value_emb(x) + self.pos_emb(x)
        x = self.dropout(x)
        return x
    
class TokenEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(TokenEmbedding, self).__init__()

        self.token_emb = nn.Embedding(1, d_model)
        self.pos_emb   = PositionalEmbedding(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb(x)
        x = self.dropout(x)
        return x