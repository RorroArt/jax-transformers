import jax
from jax import random, jit, grad
from model.bert import BERT, RobertaLMHead 
from model.train import cross_entropy, Adam
from load_roberta import checkpoint2jax

key = random.PRNGKey(0)
subkey, sub_key, k3 = random.split(key, 3)


init_bert, apply_bert = BERT(RobertaLMHead(50265, 768), 768, 12, 3072, 12, 0.2)

params = checkpoint2jax()
inputs = random.normal(sub_key, [5,10,768])
targets = random.randint(sub_key, (5,10), 0, 4)

init_adam, update_adam = Adam(3e-4)
adam_state = init_adam(params)

output = jit(apply_bert)
output_grad = output(params, inputs, k3)
loss = jax.value_and_grad(cross_entropy)

l, grads = loss(params, apply_bert, inputs, k3, targets)

params = update_adam(params, grads, adam_state)


