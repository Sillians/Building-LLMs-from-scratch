import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.output_dimension = output_dimension
        
        self.weight_query = nn.Parameter(torch.rand(input_dimension, output_dimension))
        self.weight_key   = nn.Parameter(torch.rand(input_dimension, output_dimension))
        self.weight_value = nn.Parameter(torch.rand(input_dimension, output_dimension))
        
    
    def forward(self, x: torch.Tensor):
        keys = x @ self.weight_key
        queries = x @ self.weight_query
        values = x @ self.weight_value
        
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        context_vectors = attention_weights @ values
        return context_vectors

        