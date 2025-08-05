import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# ---------------------------- Ipuput Embbeding ---------------------------------------------------

class _InputEmbbeding(nn.Module) :
    def __init__(self, dmodel:int, vocab_size:int) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.embbeding = nn.Embedding(vocab_size, dmodel)

    def forward(self, x) :
        return self.embbeding(x) * math.sqrt(self.dmodel) # multiply for normalization to overcome very small value

# ---------------------------- Positional Encoding ---------------------------------------------------

class _PositionalEncoding(nn.Module) :
    def __init__(self, dmodel:int, seq_len:int, dropout:float) -> None :
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, dmodel)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, seq_len, dmodel)

        self.register_buffer('pe',pe)

    def forward(self, x) :
        x = x + self.pe[:, :x.shape[1], :].clone().detach().requires_grad_(False)

        return self.dropout(x)
    
# ---------------------------- Multihead Attention  ---------------------------------------------------

class _MultiHeadAttention(nn.Module) :
    def __init__(self, dmodel:int, head:int, dropout:float) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.head = head
        assert dmodel % head == 0, 'dmodel is not divisible by head'
        self.dk = dmodel // head
        self.wq = nn.Linear(dmodel, dmodel)
        self.wk = nn.Linear(dmodel, dmodel)
        self.wv = nn.Linear(dmodel, dmodel)
        self.wo = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask) :
        dk = k.size(-1)
        
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(dk)

        if mask is not None :
            attention_score = attention_score.masked_fill(mask==0, -1e9)

        attention_score = F.softmax(attention_score, dim = -1) # dout

        return attention_score @ v

    def forward(self, q, k, v, mask) :
        q = self.wq(q)  # Query
        k = self.wk(k)  # Key
        v = self.wv(v)  # Value
        
        q = q.view(q.size(0), q.size(1), self.head, self.dk).transpose(1,2)
        k = k.view(k.size(0), k.size(1), self.head, self.dk).transpose(1,2)
        v = v.view(v.size(0), v.size(1), self.head, self.dk).transpose(1,2)

        x = _MultiHeadAttention.attention(q, k, v, mask)
        x = x.transpose(1,2).contiguous().view(x.size(0), -1, self.dmodel)

        return self.dropout(self.wo(x))
    
# ---------------------------- Feed Forward ---------------------------------------------------

class _FeedForwardBlock(nn.Module) :
    def __init__(self, dmodel:int, dff:int, dropout:float) -> None :
        super().__init__()
        self.linear1 = nn.Linear(dmodel, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, dmodel)
        self.relu = nn.ReLU()

    def forward(self, x) :
        return self.relu(self.linear2(self.dropout(self.linear1(x))))
    
# ---------------------------- Residual Connection ---------------------------------------------------

class _ResidualConnection(nn.Module) :
    def __init__(self, dmodel:int, dropout:float) -> None :
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dmodel)

    def forward(self, x, subLayer) :
        return x + self.dropout(subLayer(self.norm(x)))
    
# ---------------------------- Encoder Block ---------------------------------------------------

class _EncoderBlock(nn.Module) :
    def __init__(self, dmodel:int, num_head:int, dff:int, dropout:float) -> None :
        super().__init__()
        self.self_attention = _MultiHeadAttention(dmodel, num_head, dropout)
        self.feedForward = _FeedForwardBlock(dmodel, dff, dropout)
        self.residual_connection = nn.ModuleList([_ResidualConnection(dmodel, dropout) for _ in range(2)])

    def forward(self, x, src_mask) :
        x = self.residual_connection[0](x, lambda x : self.self_attention(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feedForward)

        return x
    
# ---------------------------- Encoder ---------------------------------------------------

class _Encoder(nn.Module) :
    def __init__(self,dmodel:int, num_layer:int, num_head:int, dff:int, dropout:float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_EncoderBlock(dmodel, num_head, dff, dropout) for _ in range(num_layer)])
        self.norm = nn.LayerNorm(dmodel)

    def forward(self, x, mask) :
        for layer in self.layers :
            x = layer(x, mask)
        
        return self.norm(x)
    
# ---------------------------- Decoder Block ---------------------------------------------------

class _DecoderBlock(nn.Module) :
    def __init__(self, dmodel:int, num_head:int, dff:int, dropout:float) :
        super().__init__()
        self.self_attention = _MultiHeadAttention(dmodel, num_head, dropout)
        self.cross_attention = _MultiHeadAttention(dmodel, num_head, dropout)
        self.feedForward = _FeedForwardBlock(dmodel, dff, dropout)
        self.residula_connection = nn.ModuleList([_ResidualConnection(dmodel, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask) :
        x = self.residula_connection[0](x, lambda x : self.self_attention(x, x, x, tgt_mask))
        x = self.residula_connection[1](x, lambda x : self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residula_connection[2](x, self.feedForward)

        return x

# ---------------------------- Decoder ---------------------------------------------------

class _Decoder(nn.Module) :
    def __init__(self, dmodel:int, num_layer:int, num_head:int, dff:int, dropout:float):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderBlock(dmodel, num_head, dff, dropout) for _ in range(num_layer)])
        self.norm = nn.LayerNorm(dmodel)

    def forward(self, x, encoder_output, src_mask, tgt_mask) :
        for layer in self.layers :
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

# ---------------------------- Projection Layer ---------------------------------------------------

class _ProjectionLayer(nn.Module) :
    def __init__(self, dmodel:int, vocab_size:int) :
        super().__init__()
        self.linear = nn.Linear(dmodel, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, training=True):

        if training :
            return torch.log_softmax(self.linear(x), dim=-1)
        else : 
            return self.softmax(self.linear(x))

class Transformer(nn.Module) :
    def __init__(self, dmodel:int, vocab_size:int, seq_len:int, num_layer:int, num_head:int, dff:int, dropout:float) :
        super().__init__()
        self.max_len = seq_len
        self.vocab_size = vocab_size

        self.encoder = _Encoder(dmodel, num_layer, num_head, dff, dropout)
        self.decoder = _Decoder(dmodel, num_layer, num_head, dff, dropout)

        self.embedding = _InputEmbbeding(dmodel, vocab_size)

        self.pos = _PositionalEncoding(dmodel, seq_len, dropout)
        # self.tgt_pos = _PositionalEncoding(dmodel, seq_len, dropout)

        self.projectionLayer = _ProjectionLayer(dmodel, vocab_size)

        for p in self.parameters() :
          if p.dim() > 1 :
              nn.init.xavier_uniform_(p)
        print("Weights has been Initialized")

    def encode(self, src, src_mask) :
        src = self.embedding(src)
        src = self.pos(src)

        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask) :
        tgt = self.embedding(tgt)
        tgt = self.pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def projection(self, x, training = True) :
        return self.projectionLayer(x, training)
    
    def getGreedyDecode(self, src, src_mask, device, batch_size, sos_token, eos_token) :

        decoder_input = torch.empty(batch_size, 1).fill_(sos_token).type_as(src).to(device)

        encoder_output = self.encode(src, src_mask)

        while True :
            if decoder_input.size(1) > self.max_len :
                break
            
            decoder_mask = _mask(decoder_input.size(1)).type_as(src_mask).to(device)
            
            decoder_output = self.decode(encoder_output, src_mask, decoder_input, decoder_mask)
            out = self.projection(decoder_output, False)

            next_word = torch.argmax(out, dim=-1)
            decoder_input = torch.cat([decoder_input, next_word], dim=1)

            if (next_word == eos_token).all() :
                break

        return decoder_input

# ---------------------------- Masking ---------------------------------------------------

def _mask(size) :
  return torch.triu(torch.ones(size, size), diagonal=1).bool().unsqueeze(0)
