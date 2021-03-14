import jax
import jax.numpy as jnp
import numpy as onp
from jax import random, grad, jit, vmap

from torch.utils.data import DataLoader

# ----------- Loss and optimizer -----------------

def cross_entropy(params, model, inputs, rng, target):
    outputs = model(params, inputs, rng)
    logprobs = jax.nn.log_softmax(outputs)
    nll = jnp.take_along_axis(logprobs, jnp.expand_dims(target, -1), 1)
    return -nll.mean()

def Adam(lr, betas=(0.9, 0.999), eps=1e-8):
    def init_state(params):
        i = 0
        m = jax.tree_map(jnp.zeros_like, params)
        v = jax.tree_map(jnp.zeros_like, params)
        return i, m, v
    def update(params, grads, state):
        b1, b2 = betas
        i, m, v = state

        i = i + 1
        m = jax.tree_multimap(lambda _m, g: b1*_m + (1 - b1)*g, m, grads)
        v = jax.tree_multimap(lambda _v, g: b2*_v + (1 - b2)*g, m, grads)

        m_hat = jax.tree_multimap(lambda _m: _m/(1 - b1**i), m) 
        v_hat = jax.tree_multimap(lambda _v: _v/(1 - b2**i), v)

        new_params = jax.tree_multimap(lambda param, _m, _v: param - lr*_m/(jnp.sqrt(_v) + eps), params, m_hat, v_hat)
        
        return new_params, i, m, v

    return init_state, update

# ---------------- Trainer Config ---------------------
class TrainerConfig:
    learning_rate = 3e-4
    betas = (0.9, 0.99)
    num_epochs = 10
    num_workers = 1
    batch_size = 16

    def __init__(self, **kwags):
        for k, v in kwags.items():
            setattr(self, k, v)

# --------------------- Trainer (WIP) ----------------------------
@jax.jit
def trainer(params, apply_loss, train_dataset, test_dataset, rng, config):

    def run_epoch(key, mode):
        is_training = True if mode == 'train' else False
        dataset = train_dataset if is_training else test_dataset
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        apply = jax.value_and_grad(apply_loss) if is_training else apply_loss

        for i, batch in enumerate(dataloader):
            key, subkey = random.split(key)
            x, y = batch

            output, output_grad = apply(pa   rams, x, y, subkey, mode)
            

    # Loop
    for epoch in config.num_epochs:
        key, subkey = random.split(key)


        

    return params