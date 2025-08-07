from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distrax import Categorical as Cat
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
        zn, posterior = self.vae.encode(sn, key=key2)
        dists['posterior'] = posterior
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
    _, _, sn = data
    batch_size = sn.shape[0]
    sn_hat, dists = jax.vmap(model)(data, key=jr.split(key, batch_size))
    # reconstruction error
    reconst = jnp.sum((sn - sn_hat) ** 2, axis=range(1, len(sn.shape))).mean()
    # kld ( posetrior || prior )
    posterior, prior = dists['posterior'], dists['prior']
    match (posterior, prior):
        case Gaussian(), Gaussian():
            qz = MvNormal(posterior.mean, posterior.std)
            pz = MvNormal(prior.mean, prior.std)
            kld = qz.kl_divergence(pz).mean()
        case GaussianMixture(), GaussianMixture():
            # categorical kld
            qy = Cat(posterior.logits)
            py = Cat(prior.logits)
            kld_cat = qy.kl_divergence(py)
            # gaussian kld
            qz = MvNormal(posterior.means, posterior.stds)
            pz = MvNormal(prior.means, prior.stds)
            kld_gauss = (jnp.exp(qy.logits) * qz.kl_divergence(pz)).sum(axis=-1)
            kld = (kld_cat + kld_gauss).mean()
    # compute loss + metrics
    loss = reconst + kld
    metrics = {'loss': loss, 'reconst': reconst, 'kld': kld}
    if isinstance(model.vae, GMVAE):
        metrics.update(
            {
                'kld (categorical)': kld_cat.mean(),
                'kld (gaussian)': kld_gauss.mean(),
                'entropy (posterior)': qy.entropy().mean(),
                'entropy (prior)': py.entropy().mean(),
            }
        )
    return loss, metrics
