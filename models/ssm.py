from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distrax import Categorical
from distrax import MultivariateNormalDiag as MvNormal
from jaxtyping import Array, PRNGKeyArray, PyTree

from .distributions import Distribution, Gaussian, GaussianMixture
from .gmvae import GMVAE
from .vae import VAE


class SSM(eqx.Module):
    """State-Space Model with Variation Inference.

    *Note* that one need to ensure that the encoder and transition neural networks have
    the same output dimensions, so that `self.vae.split` can be applied to the output of
    both for computing (transition) prior and posterior.
    """

    vae: GMVAE | VAE
    tr: Callable[[Array], Array]

    def __call__(
        self,
        data: tuple[Array, Array, Array],
        *,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, Distribution]]:
        """Forward a Markov transition through the state-space model.

        Args:
            data (`tuple[Array, Array, Array]`):
                A 3-tuple containing the current state, action, and next state.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A 2-tuple containing the reconstructed next state, and a dictionary of
            parameters of prior and postesrior distributions.
        """
        dists = {}
        s, a, sn = data  # state, action, next_state
        key1, key2 = jr.split(key)
        # transition (prior)
        z, _ = self.vae.encode(s, key=key1)
        dists['prior'] = self.vae.split(self.tr(jnp.concat([z, a])))
        # posterior
        zn, dists['posterior'] = self.vae.encode(sn, key=key2)
        # return
        return self.vae.decode(zn), dists


@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(
    model: SSM,
    data: tuple[Array, Array, Array],
    *,
    key: PRNGKeyArray,
) -> tuple[Array, PyTree]:
    """negative evidence lower bound (elbo)."""
    x, batch_size = data[-1], data[0].shape[0]
    x_hat, dists = jax.vmap(model)(data, key=jr.split(key, batch_size))
    # reconstruction error
    reconst = jnp.sum((x - x_hat) ** 2, axis=range(1, len(x.shape))).mean()
    # kld ( posetrior || prior )
    posterior, prior = dists['posterior'], dists['prior']
    if isinstance(posterior, Gaussian) and isinstance(prior, Gaussian):
        qz, pz = prior.to(), posterior.to()
        kld = pz.kl_divergence(qz).mean()
    if isinstance(posterior, GaussianMixture) and isinstance(prior, GaussianMixture):
        # categorical kld
        qy, py = Categorical(posterior.logits), Categorical(prior.logits)
        kld_cat = qy.kl_divergence(py)
        # gaussian kld
        qz = MvNormal(posterior.means, posterior.stds)
        pz = MvNormal(prior.means, prior.stds)
        kld_gauss = (jnp.exp(qy.logits) * qz.kl_divergence(pz)).sum(axis=-1)
        kld = (kld_cat + kld_gauss).mean()
    # compute loss + metrics
    loss = reconst + kld
    metrics = {'train/loss': loss, 'train/reconst': reconst, 'train/kld': kld}
    if isinstance(posterior, Gaussian) and isinstance(prior, Gaussian):
        pass
    if isinstance(posterior, GaussianMixture) and isinstance(prior, GaussianMixture):
        metrics = metrics | {
            'train/kld-cat': kld_cat.mean(),
            'train/kld-gauss': kld_gauss.mean(),
            'train/ent-qy': qy.entropy().mean(),
            'train/ent-py': py.entropy().mean(),
        }
    return loss, metrics
