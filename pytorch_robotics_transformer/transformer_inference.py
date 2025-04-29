import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

# This implementation is similar to tf.keras.layers.MultiHeadAttention, not torch.nn.MultiheadAttention.
# This can be used in the situation where query = key = value.
# In RT-1 we don't set value_dim. Therefore, values_dim = key_dim.
class TF_MultiHeadAttention(nn.Module):
    def __init__(self, 
                 heads: int, 
                 d_model: int, 
                 key_dim: int, 
                 value_dim: Optional[int] = None, 
                 dropout: float = 0.1,
                 return_attention_scores: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.h = heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim else key_dim # if value_dim is None, value_dim will be key_dim.
        self.return_attention_scores = return_attention_scores

        self.q_linear = nn.Linear(d_model, self.h * self.key_dim)
        self.k_linear = nn.Linear(d_model, self.h * self.key_dim)
        self.v_linear = nn.Linear(d_model, self.h * self.value_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.h * self.value_dim, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.key_dim)
        q = self.q_linear(q).view(bs, -1, self.h, self.key_dim)
        v = self.v_linear(v).view(bs, -1, self.h, self.value_dim)
        
        # transpose to get dimensions bs * h * sl * key_dim or bs * h * sl * value_dim
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2) # calculate attention using function we will define next

        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.key_dim)
        # q: (bs, h, sl, key_dim)
        # k.transpose(-2, -1) : (bs, h, key_dim, sl)
        # score: (bs, h, sl, sl)
        
        if mask is not None:
            # mask: (sl, sl)
            mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = F.softmax(scores, dim=-1)

        
        if self.dropout is not None:
            scores = self.dropout(scores)


        attention_output = torch.matmul(scores, v)
        # score: (bs, h, sl, sl)
        # v : (bs, h, sl, value_dim)
        # output: (bs, h, sl, value_dim)
        
        # concatenate heads and put through final linear layer
        concat = attention_output.transpose(1,2).contiguous().view(bs, -1, self.h * self.value_dim) # (bs, sl, heads * value_dim)
        output = self.out(concat) # (bs, sl, d_model)

        return output

# input_size and output_size: (bs, sl, feed_forward_size)
class _TransformerLayer(nn.Module):
    """A single transformer block."""
    def __init__(self,
            layer_size: int = 4096, # This corresponds to key_dim which is the size of each attention head for query, key and values.
            num_heads: int = 8,
            feed_forward_size: int = 512, # This corresponds to d_model which is embedding dimension of each token in transformer part.
            dropout_rate: float = 0.1,
            return_attention_scores: bool = False):

        super(_TransformerLayer, self).__init__()
        self._return_attention_scores = return_attention_scores

        self.norm_1 = nn.LayerNorm(feed_forward_size)
        self.attn = TF_MultiHeadAttention(num_heads,feed_forward_size, layer_size, dropout=dropout_rate, return_attention_scores=return_attention_scores)
        self.ff = nn.Linear(feed_forward_size, feed_forward_size)
        self.norm_2 = nn.LayerNorm(feed_forward_size)
        self.dropout_1 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x1 = self.norm_1(x)
        attn_results = self.attn(x1, x1, x1, mask=mask)
        x1, score = attn_results, None
        x = x + x1

        y = self.norm_2(x)
        ff_y = self.ff(y)
        ff_y = self.dropout_1(ff_y)
        x = x + ff_y

        return x, score

class Transformer(nn.Module):
    def __init__(self,
            num_layers: int = 1, # Number of transformer layers.
            layer_size: int = 4096, # This corresponds to key_dim which is the size of each attention head for query, key and values.
            num_heads: int = 8,
            feed_forward_size: int = 512, # This corresponds to d_model which is embedding dimension of each token in transformer part.
            dropout_rate: float = 0.1,
            vocab_size: int = 256, # Dimensionality of tokens from the output layer. This is also dimensionality of tokens from the input layer.
            input_token_emb_dim: int = 512, # embedding dim of input tokens.
            return_attention_scores: bool = False,
            max_seq_len: int = 256 # Maximum sequence length. This Transformer can't receive tokens that are more than this number.
            ):
        super(Transformer, self).__init__()

        # issue#3 use nn.ModuleList
        self._layers = nn.ModuleList([
        _TransformerLayer(  # pylint: disable=g-complex-comprehension
            layer_size=layer_size,
            num_heads=num_heads,
            feed_forward_size=feed_forward_size,
            dropout_rate=dropout_rate,
            return_attention_scores=return_attention_scores)
            for _ in range(num_layers)
        ])

        self._token_emb = nn.Linear(input_token_emb_dim, feed_forward_size)
        self._position_emb = nn.Embedding(max_seq_len, feed_forward_size) # <--- 工夫が必要 ここだけBERTのようにする？
        self._output_tokens = nn.Linear(feed_forward_size, vocab_size)

    # inputs: (bs, seq, emb_dim). emb_dim = vocab_size
    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        # 1. Token Embeddings
        tokens_embeddings = self._token_emb(inputs) # (bs, seq_len, feed_forward_size)

        # 2. Transformer Positional Embedding：
        position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
        position_ids = torch.tile(position_ids.unsqueeze(0), dims=(batch_size, 1)) # (bs, seq_len)
        position_embeddings = self._position_emb(position_ids) # (bs, seq_len, feed_forward_size)

        # Add the two embedded tensors together
        x = tokens_embeddings + position_embeddings # (bs, seq_len, feed_forward_size)

        scores = []

        for layer in self._layers:
            x, score = layer(x, mask=attention_mask)
            if score is not None:
                scores.append(score)
        x = self._output_tokens(x) # (bs, seq_len, vocab_size)
        return x, scores