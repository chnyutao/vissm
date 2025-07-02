from functools import partial

import equinox as eqx
import jax.random as jr
import optax
import tyro
import wandb
from tqdm.auto import tqdm

from config import Config
from dataset import make_random_walks
from models import SSM, VAE, MLPDecoder, MLPEncoder
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
vae = VAE(
    encoder=MLPEncoder(config.latent_size, key=enkey),
    decoder=MLPDecoder(config.latent_size, key=dekey),
)
tr = eqx.nn.MLP(
    config.latent_size + 4,
    config.latent_size * 2,
    width_size=128,
    depth=1,
    key=dekey,
)
model = SSM(vae=vae, tr=tr)

# init optimizer
opt = optax.adam(config.lr)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

# main loop
train_step = partial(train_step, opt=opt, callback=lambda x: wandb.log(x))
for _ in tqdm(range(config.epochs)):
    # train
    for batch in dataset:
        key, subkey = jr.split(key)
        model, opt_state = train_step(model, batch, opt_state, key=subkey)
