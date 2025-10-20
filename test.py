import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from jax.lax import stop_gradient as sg
from jaxtyping import Array, PRNGKeyArray
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import plots
from models.distributions import Categorical, Gaussian, GaussianMixture

cfgs = [
    {'alpha': 0.4, 'color': 'darkorange', 'label': 'posterior/1'},
    {'alpha': 0.8, 'color': 'lavender', 'label': 'posterior/2'},
]

# init configs
run = wandb.init(project='test')
epochs = 2000
k = 2
lr = 1e-1
ngd = True
opt_name = 'adam'
seed = 42
run.name = f'{"NGD" if ngd else "GD"}/{opt_name}-k{k}-lr={lr:.0e}-key={seed}'

# init distributions
key, *keys = jr.split(jr.key(seed), 4)
qs = [
    Gaussian(jnp.ones((2,)), jnp.ones((2,)) / 10),
    Gaussian(-jnp.ones((2,)), jnp.ones((2,)) / 10),
]
p = GaussianMixture(
    weight=Categorical(jr.normal(keys[0], (k,))),
    components=Gaussian(
        mean=jr.normal(keys[1], (k, 2)),
        std=jr.uniform(keys[2], (k, 2)),
    ),
)

# init optimizer
opt = getattr(optax, opt_name)(lr, b2=0)
opt_state = opt.init(eqx.filter(p, eqx.is_array))


# natural gradient
@jax.custom_jvp
def gaussian_ngd(p: Gaussian) -> Gaussian:
    return p


@gaussian_ngd.defjvp
def _gaussian_ngd_jvp(p, grads):
    (p,) = p
    (grads,) = grads
    grads = Gaussian(
        mean=grads.mean * (p.std**2),
        std=grads.std / 2.0 * (p.std**2),
    )
    return p, grads


# loss function
@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(
    prior: GaussianMixture,
    posterior: Gaussian,
    *,
    key: PRNGKeyArray,
) -> tuple[Array, dict[str, Array]]:
    z = posterior.sample(key=key)
    # e-step: responsibilities
    log_weights = jax.nn.log_softmax(prior.weight.logits)  # (k,)
    if ngd:
        components = gaussian_ngd(prior.components)
    else:
        components = prior.components
    log_components = components.to().log_prob(z)  # (k,)
    rho = sg(jax.nn.softmax(log_weights + log_components, axis=-1))  # (k,)
    # m-step: gradient descent
    loss = -(rho * (log_weights + log_components)).sum(axis=-1).mean()
    # # kld
    # logp = jnp.array(prior.to().log_prob(z))
    # loss = (-logp).mean()
    return loss, {
        'loss': loss,
        'entropy': prior.weight.to().entropy().mean(),
    }


# main loop
for epoch in tqdm(range(epochs)):
    # train
    key, key1, key2 = jr.split(key, 3)
    q = qs[jr.bernoulli(key1).astype(int)]
    [_, metrics], grads = loss_fn(p, q, key=key2)
    updates, opt_state = opt.update(grads, opt_state)
    p = eqx.apply_updates(p, updates)
    # eval
    wandb.log(metrics)
    if epoch % 100 != 0:
        continue
    heatmap = plots.Heatmap().show(p, qs, cfgs)
    wandb.log({'heatmap': wandb.Image(heatmap.fig)})
    plt.close('all')
