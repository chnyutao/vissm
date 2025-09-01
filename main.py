from functools import partial

import equinox as eqx
import jax.random as jr
import optax
import tyro
import wandb
from tqdm.auto import tqdm

from config import Config
from dataset import make_random_walks
from utils import eval_step, make_model, train_step

# configuration
config = tyro.cli(Config)
key = jr.key(config.seed)
wandb.init(project='vissm', config=config.asdict())

# init dataset
key, subkey = jr.split(key)
dataset = make_random_walks(
    config.data.n,
    config.data.length,
    key=subkey,
    batch_size=config.data.batch_size,
    shuffle=config.data.shuffle,
)

# init model
key, subkey = jr.split(key)
model = make_model(config, key=subkey)

# init optimizer
opt = optax.adam(config.lr)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

# main loop
train_step = partial(train_step, callback=wandb.log, opt=opt)
eval_step = partial(eval_step, callback=wandb.log)
for _ in tqdm(range(config.epochs)):
    # eval
    key, subkey = jr.split(key)
    eval_step(model, key=subkey)
    # train
    for batch in dataset:
        key, subkey = jr.split(key)
        model, opt_state = train_step(model, batch, opt_state, key=subkey)
eval_step(model, key=key)

# save model
eqx.tree_serialise_leaves('model.eqx', model)
wandb.save('model.eqx')
