"""
Attention mechanism factory for creating different attention types.
"""

from fla.layers.linear_attn import LinearAttention
from fla.layers.mamba import Mamba
from fla.layers.delta_net import DeltaNet
from setattn.setattn_legacy import SetAttention_Linear_Legacy
from setattn.setattn_linear import SetAttention_Linear

def create_attention(config, layer_index=None):
    """
    Factory function to create attention mechanism based on config.
    
    Args:
        config: Configuration object with attn attribute
        
    Returns:
        Attention module instance
        
    Raises:
        NotImplementedError: If attention type is not supported
    """
    
    if config.attn.type == 'setattn_linear_legacy':
        return SetAttention_Linear_Legacy(config)
    
    elif config.attn.type == 'setattn_linear':
        return SetAttention_Linear(config,layer_index=layer_index)
    
    elif config.attn.type == 'linear_attention':
        return LinearAttention(hidden_size=config.n_embd,feature_map=config.attn.feature_map,
                               num_heads=config.n_head)
        
    elif config.attn.type == 'mamba':
        return Mamba(hidden_size=config.n_embd,use_bias=config.bias)
    
    elif config.attn.type == 'delta_net':
        return DeltaNet(hidden_size=config.n_embd, num_heads=config.n_head)
    
    else:
        raise NotImplementedError(f"Attention type '{config.attn.type}' is not implemented")
