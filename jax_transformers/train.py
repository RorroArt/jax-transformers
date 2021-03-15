import jax
import jax.numpy as jnp
import numpy as onp
from jax import random, grad, jit, vmap

from torch.utils.data import DataLoader

# ----------- Loss and optimizer -----------------

def cross_entropy(params, model, inputs, target, rng, mode):
    outputs = model(params, inputs, rng, mode)
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
        
        return new_params, (i, m, v)

    return init_state, update

# ---------------- Trainer Config ---------------------
class TrainerConfig:
    learning_rate = 3e-4
    betas = (0.9, 0.99)
    num_epochs = 10
    num_workers = 1
    batch_size = 16
    print_every = 100

    def __init__(self, **kwags):
        for k, v in kwags.items():
            setattr(self, k, v)

# --------------------- Trainer (WIP) TODO: test ----------------------------
@jax.jit
def trainer(params, apply_loss, train_dataset, test_dataset, rng, config):

    def run_epoch(params, apply_loss, optim, optim_state, epoch, key, mode):
        is_training = True if mode == 'train' else False
        dataset = train_dataset if is_training else test_dataset
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        apply = jax.value_and_grad(apply_loss) if is_training else apply_loss
        loss_sum = 0

        for i, batch in enumerate(dataloader):
            key, subkey = random.split(key)
            x, y = batch
            
            if is_training:
                loss, grad = apply(params, x, y, subkey, mode)
                params, optim_state = optim(params, grad, optim_state)

            else:
                loss = apply(params, x, y, subkey, mode)

            if i % config.print_every == 0:
                print('Epoch: %.d - Iter: %.d - Loss: %.4f - Mode: %s' % (epoch, i, loss, mode))
            loss_sum += i.item()

        return params, optim_state, loss_sum / i
    
    init_optim, apply_optim = Adam(config.lr, config.betas)
    optim_state = init_optim(params)

    # Loop
    for epoch in config.num_epochs:
        key, subkey = random.split(key)
        params, optim_state, train_mean = run_epoch(
            params,
            apply_loss, 
            apply_optim,
            optim_state,
            epoch,
            subkey,
            'train'
        ) 
        if test_dataset is not None:
            params, optim_state, test_mean = run_epoch(
                params,
                apply_loss, 
                apply_optim,
                optim_state,
                epoch,
                subkey,
                'train'
            )
        else:
            test_mean = 0
    
        print(
            '--------------\nEpoch: %d - Training mean: %.4f - Test mean: %.4f\n------------'
            % (epoch, train_mean, test_mean)
            )
        

    return params