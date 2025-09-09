import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dimension, output_dimension, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (output_dimension % num_heads == 0), \
            "output_dimension must be divisible by num_heads"

        self.output_dimension = output_dimension
        self.num_heads = num_heads
        self.head_dim = output_dimension // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_key = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_value = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.out_proj = nn.Linear(output_dimension, output_dimension)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, input_dimension = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
        # this will result in errors in the mask creation further below. 
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forwar

        keys = self.W_key(x) # Shape: (b, num_tokens, output_dimension)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, output_dimension) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attention_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attention_scores.masked_fill_(mask_bool, -torch.inf)
        
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vector = (attention_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.output_dimension = self.num_heads * self.head_dim
        context_vector = context_vector.contiguous().view(b, num_tokens, self.output_dimension)
        context_vector = self.out_proj(context_vector) # optional projection

        return context_vector
