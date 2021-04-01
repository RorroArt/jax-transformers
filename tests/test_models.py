import os

import jax
from jax import random, jit, grad
from jax_transformers import BERT, GPT, Transformer, RobertaLMHead 

#os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/home/rodrigo/xla"

def test_bert():
    key = random.PRNGKey(0)
    k1, k2, k3 = random.split(key, 3)
    
    inputs = random.randint(k2, [5,10,4], 0, 3)
    init_bert, apply_bert = BERT(RobertaLMHead(4, 128), 4, 128, 4, 128*4, 4, 0.2)
    
    params = init_bert(k1)
    apply_jitted = jit(apply_bert)
    output = apply_jitted(params, inputs, k3)

    assert output.shape == (5, 10, 4)

def test_gpt():
    key = random.PRNGKey(0)
    k1, k2, k3 = random.split(key, 3)

    inputs = random.normal(k2, [5,10,128])
    init_gpt, apply_gpt = GPT(RobertaLMHead(4, 128), 128, 4, 128*4, 4, 0.2)

    params = init_gpt(k1)
    apply_jitted = jit(apply_gpt)
    output = apply_jitted(params, inputs, k3)

    assert output.shape == (5, 10, 4)

def test_transformer():
    key = random.PRNGKey(0)
    k1, k2, k3 = random.split(key, 3)

    encoder_inputs = random.normal(k2, [5,10,128])
    decoder_inputs = random.normal(k2, [5,10,128])
    init_transformer, apply_transformer = Transformer(RobertaLMHead(4, 128), 128, 4, 128*4, 4, 0.2)

    params = init_transformer(k1)
    apply_jitted = jit(apply_transformer)
    output = apply_jitted(params, encoder_inputs, decoder_inputs, k3)

    assert output.shape == (5, 10, 4)

