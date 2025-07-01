from functools import partial

import equinox as eqx
import jax.random as jr
import optax
import tyro
import wandb
from tqdm.auto import tqdm

from config import Config
from dataset import make_random_walks
from models import VAE, MLPDecoder, MLPEncoder
from utils import train_step

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
key, enkey, dekey = jr.split(key, 1 + 2)
model = VAE(
    encoder=MLPEncoder(config.latent_size, key=enkey),
    decoder=MLPDecoder(config.latent_size, key=dekey),
)

# init optimizer
opt = optax.adam(config.lr)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

# main loop
train_step = partial(train_step, opt=opt)
for _ in tqdm(range(config.epochs)):
    # train
    for batch in dataset:
        key, subkey = jr.split(key)
        model, opt_state, metrics = train_step(model, batch, opt_state, key=subkey)
        wandb.log(metrics)
