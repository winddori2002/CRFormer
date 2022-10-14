import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers import *

class CRFormer(nn.Module):
    """
    CRFormer:
        input_size  :  inital size (1).
        hidden_size :  hidden size.
        num_layers  : number of layers for Transformer encoder (default:6)
        d_ffn       : hidden size for Transformer ffn (default:256)
        n_head      : number of head for multi-head attention (default:4)
        d_k         : dimension for each head in multi-head attention (default:64)
        dropout     : dropout ratio (default:0.3)
        use_feat    : usage feature option for CRFormer-F (default:False)
        features    : feature list of claim data 
        
    feat_emb : feature embedding for CRFormer-F
    enc_emb  : Nested sequence embedding
    encoder  : Transformer encoder
    decoder  : decoder with context attention
    """
    max_length = 61
    def __init__(self, args):
        super(CRFormer, self).__init__()
        
        d_model          = args.hidden_size
        drop_ratio       = 0.4 * (1 + (args.sequence_length - 6) / 10) # length-wise dropout ratio
        self.use_feat    = args.use_feat
        self.device      = args.device
        self.output_size = self.max_length - args.sequence_length 
        
        if args.use_feat:
            self.feat_emb = FeatureEmbedding(args.f_sizes, d_model, args.dropout) # feature embedding
            
        self.enc_emb = NestedSequenceEmbedding(args.input_size, d_model, drop_ratio)
        self.encoder = Encoder(d_model, args.d_ffn, args.n_head, args.d_k, args.num_layers, 'relu', args.dropout)
        self.decoder = Decoder(d_model, self.output_size, args.dropout)
        
    def _set_token(self, value, size):
        return torch.tensor(value).repeat(size).to(self.device)
        
    def forward(self, x, features=None):
        """
        Inputs:
            x: Nested sequence (B, L1, L2) - L1 is input sequence length of main reliability, L2 is sub sequence length of sub reliability
            features: Feature list (B, num features)
        Returns:
            y: Prediction outputs (B, max_length - L1) - Predictions for future failures
        """
        
        x = self.enc_emb(x) # (B, L1, L2) -> (B, L1, C)
        if self.use_feat:
            features = self.feat_emb(features) # (B, num features) -> (B, C)

        enc_out, _ = self.encoder(x, mask=None) # (B, L1, C)
        output     = self.decoder(enc_out, features) 
        
        return output

class Encoder(nn.Module):
    def __init__(self, d_model, d_ffn, n_head, d_k, num_layers, activation='relu', dropout=0.1):
        super(Encoder, self).__init__()
        
        self.encoder = nn.ModuleList([EncoderLayer(d_model, d_ffn, n_head, d_k, activation, dropout) for i in range(num_layers)])
        
    def forward(self, x, mask=None, return_attns=False):
        
        attn_list = []
        for encoder_layer in self.encoder:
            x, attn   = encoder_layer(x, mask=mask)
            attn_list += [attn] if return_attns else []
        
        return x, attn_list
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_head, d_k, activation='relu', dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mh_attn  = MultiHeadAttention(d_model, n_head, d_k, dropout)
        self.pos_conv = PositionwiseConvLayer(d_model, d_ffn, activation, dropout)

    def forward(self, x, mask=None):
        
        x, attn  = self.mh_attn(x, x, x, mask=mask)
        x        = self.pos_conv(x)
        
        return x, attn
    
class Decoder(nn.Module):
    def __init__(self, d_model, output_size, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.projection = nn.Linear(d_model, d_model)
        self.c_vector   = nn.Parameter(torch.randn(d_model,1))
        self.norm       = nn.LayerNorm(d_model)
        self.linear     = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, output_size))
    
    def forward(self, x, features):
        
        x    = self.projection(x)
        attn = torch.matmul(x, self.c_vector)
        attn = F.softmax(attn, dim=1)
        x    = x * attn
        
        if features is None:
            x = x.mean(dim=1)
        else:
            x = x.mean(dim=1) + features # feature merging step for CRFormer-F
            x = self.norm(x)
            
        x = F.relu(self.linear(x))
        
        return x