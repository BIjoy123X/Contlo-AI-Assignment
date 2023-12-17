# Rotary Positional Embedding Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_size, max_seq_len=512):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embedding_dim = embed_size
        self.max_seq_len = max_seq_len
        self.freqs = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        self.phase = torch.zeros(embed_size)

    def forward(self, x):
        position = torch.arange(0, x.size(1)).float().to(x.device)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_emb = torch.zeros(1, x.size(1), self.embedding_dim).to(x.device)
        pos_emb[0, :, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        return x + pos_emb

# Modify GPT2Layer to use RotaryPositionalEmbedding
class GPT2Layer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout=0.1):
        super(GPT2Layer, self).__init__()
        self.multiheadattention = MultiHeadAttention(embed_size, heads)
        self.feedforward = FeedForward(embed_size, ff_hidden_dim)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.rotary_positional_embedding = RotaryPositionalEmbedding(embed_size)

    def forward(self, value, key, query, mask):
        query = self.rotary_positional_embedding(query)
        attention = self.multiheadattention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.layernorm1(attention + query)
        x = self.dropout(x)

        # Feedforward part
        forward = self.feedforward(x)

        # Again add skip connection
        out = self.layernorm2(forward + x)
        out = self.dropout(out)
        return out
    

# Group Query Attention Implementation 

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(GroupQueryAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.group_projection = nn.Linear(embed_size, embed_size)
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Project query to group dimension
        group_query = self.group_projection(query)

        # Apply multi-head attention
        group_attention = self.attention(values, keys, group_query, mask)

        # Combine group attention results
        group_attention = group_attention.view(group_attention.size(0), -1)

        # Final linear projection
        out = self.fc_out(group_attention)
        return out

# Modify GPT2Layer to use GroupQueryAttention
class GPT2Layer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout=0.1):
        super(GPT2Layer, self).__init__()
        self.group_query_attention = GroupQueryAttention(embed_size, heads)
        

    def forward(self, value, key, query, mask):
        query = self.group_query_attention(value, key, query, mask)
        return out


# Sliding Window Attention Implementation

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        super(SlidingWindowAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.window_size = window_size

        self.attention = MultiHeadAttention(embed_size, heads)
        self.avg_pool = nn.AvgPool1d(window_size, stride=1, padding=window_size // 2)

    def forward(self, values, keys, query, mask):
        attention = self.attention(values, keys, query, mask)
        # Apply average pooling along the sequence dimension
        attention = attention.transpose(1, 2)  
        attention = self.avg_pool(attention)
        attention = attention.transpose(1, 2)  
        return attention

# Modify GPT2Layer to use SlidingWindowAttention
class GPT2Layer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout=0.1):
        super(GPT2Layer, self).__init__()
        self.sliding_window_attention = SlidingWindowAttention(embed_size, heads, window_size=5)
      
    def forward(self, value, key, query, mask):
        attention = self.sliding_window_attention(value, key, query, mask)
        return out