import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.nn import softmax, gelu
from jax.nn.initializers import glorot_normal, normal, zeros, ones

# ------------- Functions ------------------
def dot_product_attn(q, k, v, dk, mask):
    attn_score = q @ k.transpose((0, 1, 3, 2))
    if mask is not None:
        attn_score = jnp.where(mask[:,:,:q.shape[-2], :q.shape[-2]] == 0, attn_score, float("-inf"))
    return softmax((attn_score) / jnp.sqrt(dk)) @ v

        
# ------------- Layers ---------------------
# Series of stax like layers

# Linear
def Linear(in_features, out_features, w_initializer=glorot_normal(), b_initializer=normal()):
    def init_params(rng):
        k1, k2 = random.split(rng)
        w, b = w_initializer(k1, (in_features, out_features)), b_initializer(k2, (out_features,))
        return (w, b)
    def apply_fn(params, inputs):
        w, b = params
        return (inputs@w) + b
    return init_params, apply_fn

# Layer Normalizaion 
def LayerNorm(in_features, eps=1e-05, beta_initializer=zeros, gamma_initializer=ones):
    def init_params(rng):
        k1, k2 = random.split(rng)
        beta, gamma = beta_initializer(k1, (in_features)), gamma_initializer(k2, (in_features))
        return (beta, gamma)
    def apply_fn(params, inputs):
        beta, gamma = params
        out = ((inputs - inputs.mean(-1, keepdims=True)) / jnp.sqrt( inputs.var(keepdims=True) + eps )) 
        return out * gamma + beta
    return init_params, apply_fn

# Dropout
def Dropout(rate):
    def apply_fn(inputs, rng, mode='train'):
        if mode == 'train':
            dropout = random.bernoulli(rng, rate, inputs.shape)
            return jnp.where(dropout, inputs / rate, 0)
        else:
            return inputs
    return apply_fn

# Embeddings 
def Embeddings(num_embeddings, embed_dim, w_initializer=normal()):
	def init_params(rng):
		k1, k2 = random.split(rng)
		w = w_initializer(k1, (num_embeddings, embed_dim)) 
		return w
	def apply_fn(params, inputs):	
		w = params
		return inputs@w
	
	return init_params, apply_fn

# Positional Embeddings
def PosEmbeddings(num_embeddings, embed_dim):
	init_embeddings, apply_embeddings = Embeddings(num_embeddings, embed_dim)

	def init_params(rng):
		k1, k2 = random.split(rng)
		return init_embeddings(k2)
	
	def apply_fn(params, inputs):
		embedding_params = params

		seq_len = inputs.shape[1]
		pos = jnp.expand_dims(jnp.arange(seq_len), 1)
		index = jnp.expand_dims(jnp.arange(embed_dim), 0)
		angles = 1 / jnp.power(1000, (2 *(index//2)) / embed_dim)
		pos_encoding  = pos * angles

		pos_encoding.at[:, 0::2].set(jnp.sin(pos_encoding[:, 0::2]))
		pos_encoding.at[:, 1::2].set(jnp.cos(angles[:, 1::2]))

		pos_encoding = jnp.expand_dims(pos_encoding, 0)
		
		out = apply_embeddings(embedding_params, inputs)
		out = out + pos_encoding

		return out
	
	return init_params, apply_fn

# Multi Head Attention
def MultiHeadAttn(embed_dim, num_heads, masked=False, block_size=1024, dropout=0):
    init_q, apply_q = Linear(embed_dim, embed_dim)
    init_k, apply_k = Linear(embed_dim, embed_dim)
    init_v, apply_v = Linear(embed_dim, embed_dim)
    init_out, apply_out = Linear(embed_dim, embed_dim)
    
    attn_dropout = Dropout(dropout)
    out_dropout = Dropout(dropout)
    
    mask = None
    if masked:
        mask = jnp.tril(jnp.ones((block_size, block_size))).reshape(1, 1, block_size, block_size)
    
    def init_params(rng):
        k1, k2, k3, k4 = random.split(rng, 4)
        return (init_q(k1), init_k(k2), init_v(k3), init_out(k4))

    def apply_fn(params, inputs, rng, mode,):
        
        k1, k2 = random.split(rng, 2)

        if type(inputs) == tuple:
            x_q, x_k, x_v = inputs
        else: 
            x_q = inputs
            x_k = inputs
            x_v = inputs


        batch, seq, E = x_k.shape
        dk = E // num_heads
        
        q_params, k_params, v_params, out_params = params
       
        q = apply_q(q_params, x_q).reshape(batch, seq, num_heads, dk).swapaxes(1,2)
        k = apply_k(k_params, x_k).reshape(batch, seq, num_heads, dk).swapaxes(1,2)
        v = apply_v(v_params, x_v).reshape(batch, seq, num_heads, dk).swapaxes(1,2)
        
        attn = dot_product_attn(q, k, v, dk, mask).reshape(batch, seq, E)
        attn = attn_dropout(attn, k1, mode)
        out = apply_out(out_params, attn)
        out = out_dropout(out, k2, mode)

        return out

    return init_params, apply_fn

# ------------ Blocks --------------------
# Transformer Encoder Block
def EncoderBlock(d_model, dff, num_heads, dropout):
    init_attn, apply_attn = MultiHeadAttn(d_model, num_heads, dropout=dropout)
    init_norm1, apply_norm1 = LayerNorm(d_model)
    init_ffn1, apply_ffn1 = Linear(d_model, dff)
    init_ffn2, apply_ffn2 = Linear(dff, d_model)
    init_norm2, apply_norm2 = LayerNorm(d_model)
    apply_dropout = Dropout(dropout)

    def init_params(rng):
        k1, k2, k3, k4, k5 = random.split(rng, 5)
        return (init_attn(k1), init_norm1(k2), init_ffn1(k3), init_ffn2(k4), init_norm2(k5))

    def apply_fn(params, inputs, rng, mode):
        
        k1, k2 = random.split(rng, 2)
        
        attn_params, norm1_params, ffn1_params, ffn2_params, norm2_params = params
        
        out = inputs + apply_attn(attn_params, inputs, k1, mode)
        out = apply_norm1(norm1_params, out)
        out_ffn1 = gelu(apply_ffn1(ffn1_params, out)) 
        out = out + apply_dropout(apply_ffn2(ffn2_params, out_ffn1), k2, mode)
        out = apply_norm2(norm2_params, out)

        return out
    
    return init_params, apply_fn

# GPT like Transformer decoder block
def GPTDecoderBlock(d_model, dff, num_heads, dropout):
    init_attn, apply_attn = MultiHeadAttn(d_model, num_heads, masked=True, dropout=dropout)
    init_norm1, apply_norm1 = LayerNorm(d_model)
    init_ffn1, apply_ffn1 = Linear(d_model, dff)
    init_ffn2, apply_ffn2 = Linear(dff, d_model)
    init_norm2, apply_norm2 = LayerNorm(d_model)
    apply_dropout = Dropout(dropout)

    def init_params(rng):
        k1, k2, k3, k4, k5 = random.split(rng, 5)
        return (init_attn(k1), init_norm1(k2), init_ffn1(k3), init_ffn2(k4), init_norm2(k5))

    def apply_fn(params, inputs, rng, mode):
        
        k1, k2 = random.split(rng, 2)
        
        attn_params, norm1_params, ffn1_params, ffn2_params, norm2_params = params
        
        out = inputs + apply_attn(attn_params, inputs, k1, mode)
        out = apply_norm1(norm1_params, out)
        out_ffn1 = gelu(apply_ffn1(ffn1_params, out)) 
        out = out + apply_dropout(apply_ffn2(ffn2_params, out_ffn1), k2, mode)
        out = apply_norm2(norm2_params, out)

        return out
    
    return init_params, apply_fn

# Full Transformer decoder block
def DecoderBlock(d_model, dff, num_heads, dropout):
    init_attn, apply_attn = MultiHeadAttn(d_model, num_heads, masked=True, dropout=dropout)
    init_norm1, apply_norm1 = LayerNorm(d_model)
    init_attn2, apply_attn2 = MultiHeadAttn(d_model, num_heads, masked=True, dropout=dropout)
    init_norm2, apply_norm2 = LayerNorm(d_model)
    init_ffn1, apply_ffn1 = Linear(d_model, dff)
    init_ffn2, apply_ffn2 = Linear(dff, d_model)
    init_norm3, apply_norm3 = LayerNorm(d_model)
    apply_dropout = Dropout(dropout)

    def init_params(rng):
        k1, k2, k3, k4, k5, k6, k7 = random.split(rng, 7)
        return (
                init_attn(k1), 
                init_norm1(k2), 
                init_attn2(k3),
                init_norm2(k4),
                init_ffn1(k5), 
                init_ffn2(k6), 
                init_norm3(k7)
                )

    def apply_fn(params, inputs, encoder_outputs, rng, mode):
        k1, k2, k3 = random.split(rng, 3)

        attn_params, norm1_params, attn2_params, norm2_params, ffn1_params, ffn2_params, norm3_params = params
        
        out = inputs + apply_attn(attn_params, inputs, k1, mode)
        out = apply_norm1(norm1_params, out)
        out = out + apply_attn2(attn2_params, (out, encoder_outputs, encoder_outputs), k2, mode)
        out = apply_norm2(norm2_params, out)
        out_ffn1 = gelu(apply_ffn1(ffn1_params, out)) 
        out = out + apply_dropout(apply_ffn2(ffn2_params, out_ffn1), k3, mode)
        out = apply_norm3(norm3_params, out)

        return out
    
    return init_params, apply_fn

# --------------- Encoder ----------------------
def TransformerEncoder(d_model, num_layers, dff, num_heads, dropout):
    init_layers = []
    apply_layers = []

    for _ in range(num_layers):
        init_layer, apply_layer = EncoderBlock(d_model, dff, num_heads, dropout)
        init_layers.append(init_layer)
        apply_layers.append(apply_layer)

    def init_params(rng):
        keys = random.split(rng, num_layers)
        return [init_layer(key) for key, init_layer in zip(keys, init_layers)]

    def apply_fn(params, inputs, rng, mode='train'):
        keys = random.split(rng, num_layers)

        layers_params =  params
        out = inputs
        for layer_params, apply_layer, key in zip(layers_params, apply_layers, keys):
            out = apply_layer(layer_params, out, key, mode)
       
        return out

    return init_params, apply_fn


# --------------- Decoder ----------------------
def TransformerDecoder(d_model, num_layers, dff, num_heads, dropout):
    init_layers = []
    apply_layers = []

    for _ in range(num_layers):
        init_layer, apply_layer = DecoderBlock(d_model, dff, num_heads, dropout)
        init_layers.append(init_layer)
        apply_layers.append(apply_layer)

    def init_params(rng):
        keys = random.split(rng, num_layers)
        return [init_layer(key) for key, init_layer in zip(keys, init_layers)]

    def apply_fn(params, inputs, encoder_outputs, rng, mode='train'):
        keys = random.split(rng, num_layers)

        layers_params =  params
        out = inputs
        for layer_params, apply_layer, key in zip(layers_params, apply_layers, keys):
            out = apply_layer(layer_params, out, encoder_outputs, key, mode)
       
        return out

    return init_params, apply_fn

# --------------- GPT like decoder ---------------------
def GPTDecoder(d_model, num_layers, dff, num_heads, dropout):
    init_layers = []
    apply_layers = []


    for _ in range(num_layers):
        init_layer, apply_layer = GPTDecoderBlock(d_model, dff, num_heads, dropout)
        init_layers.append(init_layer)
        apply_layers.append(apply_layer)

    def init_params(rng):
        keys = random.split(rng, num_layers)
        return [init_layer(key) for key, init_layer in zip(keys, init_layers)]

    def apply_fn(params, inputs, rng, mode='train'):
        keys = random.split(rng, num_layers)
        
        layers_params =  params
        out = inputs
        for layer_params, apply_layer, key in zip(layers_params, apply_layers, keys):
            out = apply_layer(layer_params, out, key, mode)
       
        return out

    return init_params, apply_fn



# --------------- GPT ----------------------
def GPT(output_head, d_model, num_layers, dff, num_heads, dropout):
    init_decoder, apply_decoder = GPTDecoder(d_model, num_layers, dff, num_heads, dropout) 
    init_output_head, apply_output_head = output_head
    
    def init_params(rng):
        k1, k2 = random.split(rng, 2)
        return init_decoder(k1), init_output_head(k2)

    def apply_fn(params, inputs, rng, mode='train'):
        decoder_params, output_head_params =  params
        out = apply_decoder(decoder_params, inputs, rng, mode) 
        out = apply_output_head(output_head_params, out)

        return out

    return init_params, apply_fn


# --------------- BERT ----------------------
def BERT(output_head, vocab_size, d_model, num_layers, dff, num_heads, dropout):
	init_embeddings, apply_embeddings = PosEmbeddings(vocab_size, d_model)
	init_encoder, apply_encoder = TransformerEncoder(d_model, num_layers, dff, num_heads, dropout) 
	init_output_head, apply_output_head = output_head
    
	def init_params(rng):
		k1, k2, k3 = random.split(rng, 3)
		return init_embeddings(k1), init_encoder(k2), init_output_head(k3)

	def apply_fn(params, inputs, rng, mode='train'):
		embedding_params, encoder_params, output_head_params = params
		embedded_inputs = apply_embeddings(embedding_params, inputs)
		out = apply_encoder(encoder_params, embedded_inputs, rng, mode)
		out = apply_output_head(output_head_params, out)
		return out

	return init_params, apply_fn


# --------------- Transformer ----------------------
def Transformer(output_head, d_model, num_layers, dff, num_heads, dropout):
    init_encoder, apply_encoder = TransformerEncoder(d_model, num_layers, dff, num_heads, dropout) 
    init_decoder, apply_decoder = TransformerDecoder(d_model, num_layers, dff, num_heads, dropout)
    init_output_head, apply_output_head = output_head
    
    def init_params(rng):
        k1, k2, k3 = random.split(rng, 3)
        return init_encoder(k1), init_decoder(k2), init_output_head(k3)

    def apply_fn(params, encoder_inputs, decoder_inputs, rng, mode='train'):
        k1, k2 = random.split(rng, 2)
        encoder_params, decoder_params, output_head_params = params
        out = apply_encoder(encoder_params, encoder_inputs, k1, mode)
        out = apply_decoder(decoder_params, decoder_inputs, out, k2, mode)
        out = apply_output_head(output_head_params, out)

        return out

    return init_params, apply_fn

#----------------- Heads ----------------------------------
# RoBERTa LM Head 
def RobertaLMHead(output_size, d_model):
    init_ffn1, apply_ffn1 = Linear(d_model, d_model)
    init_norm, apply_norm = LayerNorm(d_model)
    init_ffn2, apply_ffn2 = Linear(d_model, output_size)

    def init_params(rng):
        k1, k2, k3 = random.split(rng, 3)
        return (init_ffn1(k1), init_norm(k2), init_ffn2(k3))

    def apply_fn(params, inputs):
        ffn1_params, norm_params, ffn2_params = params
        out = gelu(apply_ffn1(ffn1_params, inputs))
        out = apply_norm(norm_params, out)
        out = apply_ffn2(ffn2_params, out)

        return out

    return init_params, apply_fn
    






