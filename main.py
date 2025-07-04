from functools import partial

import equinox as eqx
import jax.random as jr
import optax
import tyro
import wandb
from tqdm.auto import tqdm

from config import Config
from dataset import make_random_walks
from models import GMVAE, SSM, VAE, MLPDecoder, MLPEncoder
from utils import eval_step, train_step

# configuration
config = tyro.cli(Config)
key = jr.key(config.seed)
wandb.init(project='random-walk', config=config.asdict())

# init dataset
key, subkey = jr.split(key)
dataset = make_random_walks(
    config.n,
    config.length,
    key=subkey,
    batch_size=config.batch_size,
    shuffle=config.shuffle,
)

# init model
key, enkey, dekey, trkey = jr.split(key, 1 + 3)
if config.vae == 'gmvae':
    distribution_size = config.k + config.k * config.latent_size * 2
    vae = GMVAE(
        encoder=MLPEncoder(distribution_size, key=enkey),
        decoder=MLPDecoder(config.latent_size, key=dekey),
        k=config.k,
        tau=config.tau,
    )
elif config.vae == 'vae':
    distribution_size = config.latent_size * 2
    vae = VAE(
        encoder=MLPEncoder(config.latent_size * 2, key=enkey),
        decoder=MLPDecoder(config.latent_size, key=dekey),
    )
tr = eqx.nn.MLP(
    config.latent_size + 4,
    distribution_size,
    width_size=128,
    depth=1,
    key=trkey,
)
model = SSM(vae=vae, tr=tr)

# init optimizer
opt = optax.adam(config.lr)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

# main loop
train_step = partial(train_step, opt=opt, callback=lambda x: wandb.log(x))
eval_step = partial(eval_step, callback=lambda x: wandb.log(x))
for _ in tqdm(range(config.epochs)):
    # train
    for batch in dataset:
        key, subkey = jr.split(key)
        model, opt_state = train_step(model, batch, opt_state, key=subkey)
eval_step(model)
