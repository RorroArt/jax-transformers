import torch

import jax
import jax.numpy as jnp

import numpy as onp

device = torch.device('cpu')

# ------------ Covert State Dict ----------------------------
def state_dict2jax(model_dict, n_layers=12, d_model=768):
    layers = []

    # Roberta
    for i in range(n_layers):
        layer_name = 'decoder.sentence_encoder.layers.{}.'.format(i)

        # attention paramaters
        q_w = jnp.array(model_dict[layer_name+'self_attn.in_proj_weight'].detach().numpy()[:d_model].T)
        q_b = jnp.array(model_dict[layer_name+'self_attn.in_proj_bias'].numpy()[:d_model])

        k_w = jnp.array(model_dict[layer_name+'self_attn.in_proj_weight'].numpy()[d_model:d_model*2].T)
        k_b = jnp.array(model_dict[layer_name+'self_attn.in_proj_bias'].numpy()[d_model:d_model*2])
        
        v_w = jnp.array(model_dict[layer_name+'self_attn.in_proj_weight'].numpy()[d_model*2:d_model*3].T)
        v_b = jnp.array(model_dict[layer_name+'self_attn.in_proj_bias'].numpy()[d_model*2:d_model*3])
        
        attn_out_w = jnp.array(model_dict[layer_name+'self_attn.out_proj.weight'].numpy().T)
        attn_out_b = jnp.array(model_dict[layer_name+'self_attn.out_proj.bias'].numpy())
        attn_params = ((q_w, q_b), (k_w, k_b), (v_w, v_b), (attn_out_w, attn_out_b))

        # layer norm 1 parameters
        beta1 = jnp.array(model_dict[layer_name+'self_attn_layer_norm.bias'].numpy())
        gamma1 = jnp.array(model_dict[layer_name+'self_attn_layer_norm.weight'].numpy())
        layer_norm1_params = (beta1, gamma1)

        # FF1 parameters
        w1 = jnp.array(model_dict[layer_name+'fc1.weight'].numpy().T)  
        b1 = jnp.array(model_dict[layer_name+'fc1.bias'].numpy())
        ffn1_params = (w1, b1)

        # FF2 parameters
        w2 = jnp.array(model_dict[layer_name+'fc2.weight'].numpy().T)
        b2 = jnp.array(model_dict[layer_name+'fc2.bias'].numpy()) 
        ffn2_params = (w2, b2)

        # layer norm 2 parameters
        beta2 = jnp.array(model_dict[layer_name+'final_layer_norm.bias'].numpy())
        gamma2 = jnp.array(model_dict[layer_name+'final_layer_norm.weight'].numpy())
        layer_norm2_params = (beta2, gamma2)

        layers.append((attn_params, layer_norm1_params, ffn1_params, ffn2_params, layer_norm2_params))
    
    # LM head
    layer_name = 'decoder.lm_head.'

    # Linear 1
    linear1_w = jnp.array(model_dict[layer_name+'dense.weight'].numpy().T)
    linear1_b = jnp.array(model_dict[layer_name+'dense.bias'].numpy())
    linear1_params = (linear1_w, linear1_b)

    # Layer Norm
    final_beta = jnp.array(model_dict[layer_name+'layer_norm.bias'].numpy())
    final_gamma = jnp.array(model_dict[layer_name+'layer_norm.weight'].numpy())
    final_layer_norm = (final_beta, final_gamma)

    # Linear 2 
    linear2_w = jnp.array(model_dict[layer_name+'weight'].numpy())
    linear2_b = jnp.array(model_dict[layer_name+'bias'].numpy())
    linear2_params = (linear2_w, linear2_b)

    # Head params
    head_params = (linear1_params, final_layer_norm, linear2_params)
    return (layers, head_params)

# ---------------------- Convert checkpoint -------------------------
def checkpoint2jax(model_dir='models/roberta.base/model.pt'):
    device = torch.device('cpu')
    
    checkpoint = torch.load(model_dir, map_location=device) 
    params = state_dict2jax(checkpoint['model'])

    return params

