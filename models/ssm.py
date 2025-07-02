from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distrax import MultivariateNormalDiag as MvNormal
from jaxtyping import Array, PRNGKeyArray, PyTree

from .vae import VAE


class SSM(eqx.Module):
    vae: VAE
    tr: Callable[[Array], Array]

    def __call__(
        self,
        data: tuple[Array, Array, Array],
        *,
        key: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
        dists = {}
        s, a, sn = data
        key1, key2 = jr.split(key)
        # transition (prior)
        z, _ = self.vae.encode(s, key=key1)
        mean, log_std = jnp.split(self.tr(jnp.concat((z, a))), 2)
        dists['prior'] = (mean, jnp.exp(log_std))
        # posterior
        zn, posterior = self.vae.encode(sn, key=key2)
        dists['posterior'] = posterior
        # return
        return self.vae.decoder(zn), dists


def loss_fn(
    model: SSM,
    data: tuple[Array, Array, Array],
    *,
    key: PRNGKeyArray,
) -> tuple[Array, PyTree]:
    """negative evidence lower bound (elbo)."""
    _, _, sn = data
    batch_size = sn.shape[0]
    sn_hat, dists = jax.vmap(model)(data, key=jr.split(key, batch_size))
    # reconstruction error
    reconst = jnp.sum((sn - sn_hat) ** 2, axis=range(1, len(sn.shape))).mean()
    # gaussian kld(q(z|x) || p(z))
    posteriors, priors = dists['posterior'], dists['prior']
    kld = MvNormal(*posteriors).kl_divergence(MvNormal(*priors)).mean()
    # return
    loss = reconst + kld
    return loss, {'loss': loss, 'reconst': reconst, 'kld': kld}
