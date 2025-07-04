from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distrax import MultivariateNormalDiag as MvNormal
from jaxtyping import Array, PRNGKeyArray, PyTree

from .vae import VAE


class SSM(eqx.Module):
    """State-Space Model with Variation Inference.

    *Note* that we have to ensure that the encoder and the transition NNs have the same
    output dimensions, so that `self.vae.distributions` can be applied to the output of
    both for computing (transition) prior and posterior.
    """

    vae: VAE
    tr: Callable[[Array], Array]

    def __call__(
        self,
        data: tuple[Array, Array, Array],
        *,
        key: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
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
        dists['prior'] = self.vae.distribution(self.tr(jnp.concat([z, a])))
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
    # gaussian kld(q(z|x) || p(z))
    posteriors, priors = dists['posterior'], dists['prior']
    if isinstance(model.vae, VAE):
        kld = (
            MvNormal(posteriors['mean'], posteriors['std'])
            .kl_divergence(MvNormal(priors['mean'], priors['std']))
            .mean()
        )
    # return
    loss = reconst + kld
    return loss, {'loss': loss, 'reconst': reconst, 'kld': kld}
