import torch
import torch.nn as nn

class CausalAttention(nn.Module):

    def __init__(self, input_dimension, output_dimension, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.output_dimension = output_dimension
    
        self.W_query = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_key   = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_value = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, input_dimension = x.shape # New batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forward method. 
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attention_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights) # New

        context_vector = attention_weights @ values
        return context_vector
