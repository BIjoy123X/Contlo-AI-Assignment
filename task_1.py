import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GPT2Layer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout=0.1):
        super(GPT2Layer, self).__init__()
        self.multiheadattention = MultiHeadAttention(embed_size, heads)
        self.feedforward = FeedForward(embed_size, ff_hidden_dim)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.multiheadattention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.layernorm1(attention + query)
        x = self.dropout(x)

        # Feedforward part
        forward = self.feedforward(x)

        # Again add skip connection, run through normalization and finally dropout
        out = self.layernorm2(forward + x)
        out = self.dropout(out)
        return out


class GPT2(nn.Module):
    def __init__(
        self,
        embed_size=768,
        heads=12,
        num_layers=12,
        ff_hidden_dim=3072,
        dropout=0.1,
    ):
        super(GPT2, self).__init__()
        self.layers = nn.ModuleList(
            [
                GPT2Layer(embed_size, heads, ff_hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        x = self.fc(x)
        return x