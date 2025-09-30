import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from jaxtyping import Array, PRNGKeyArray
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import plots
from models.distributions import Categorical, Gaussian, GaussianMixture

wandb.init(project='test')

cfgs = [
    {'alpha': 0.4, 'color': 'darkorange', 'label': 'posterior/1'},
    {'alpha': 0.8, 'color': 'lavender', 'label': 'posterior/2'},
]
epochs = 3000
key = jr.key(0)
lr = 1e-3
n_samples = 1

# init distributions
key, subkey = jr.split(key)
qs = [
    Gaussian(jnp.ones((2,)), jnp.ones((2,)) / 10),
    Gaussian(-jnp.ones((2,)), jnp.ones((2,)) / 10),
]
p = GaussianMixture(
    weight=Categorical(jnp.ones((2,))),
    components=Gaussian(
        mean=jr.normal(subkey, (2, 2)),
        std=jnp.ones((2, 2)),
    ),
)

# init optimizer
opt = optax.sgd(lr)
opt_state = opt.init(eqx.filter(p, eqx.is_array))


# loss function
@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(
    prior: GaussianMixture,
    posterior: Gaussian,
    *,
    key: PRNGKeyArray,
) -> tuple[Array, dict[str, Array]]:
    z = jax.vmap(posterior.sample)(key=jr.split(key, n_samples))
    logp = jnp.array(prior.to().log_prob(z))
    logq = jnp.array(posterior.to().log_prob(z))
    kld = (logq - logp).mean()
    return kld, {
        'loss': kld,
        'entropy': prior.weight.to().entropy().mean(),
    }


# main loop
for epoch in tqdm(range(epochs)):
    # train
    key, subkey = jr.split(key)
    q = qs[jr.bernoulli(subkey, p=0.7).astype(int)]
    [_, metrics], grads = loss_fn(p, q, key=subkey)
    updates, opt_state = opt.update(grads, opt_state)
    p = eqx.apply_updates(p, updates)
    # eval
    wandb.log(metrics)
    if epoch % 100 != 0:
        continue
    heatmap = plots.Heatmap().show(p, qs, cfgs)
    wandb.log({'heatmap': wandb.Image(heatmap.fig)})
    plt.close('all')
