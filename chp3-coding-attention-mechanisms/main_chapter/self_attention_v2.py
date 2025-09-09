import torch
import torch.nn as nn   

class SelfAttention_v2(nn.Module):
    def __init__(self, input_dimension, output_dimension, qkv_bias=False):
        super().__init__()
        self.output_dimension = output_dimension
        
        self.weight_query = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.weight_key   = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.weight_value = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        
    def forward(self, x: torch.Tensor):
        queries = self.weight_query(x)
        keys = self.weight_key(x)
        values = self.weight_value(x)
        
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        context_vector = attention_weights @ values
        return context_vector