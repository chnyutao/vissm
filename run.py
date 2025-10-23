from functools import partial

import equinox as eqx
import jax.random as jr
import tyro
import wandb
from tqdm.auto import tqdm

from config import default_configs
from utils import eval_step, make_dataset, make_model, make_opt, train_step

# configuration
config = tyro.extras.overridable_config_cli(default_configs)
key = jr.key(config.seed)
wandb.init(project='ngem', config=config.asdict())

# initialization
key, dkey, mkey = jr.split(key, 3)
dataset = make_dataset(config, key=dkey)
model = make_model(config, key=mkey)
opt, opt_state = make_opt(config, model)

# main loop
train_step = partial(train_step, callback=wandb.log, opt=opt)
eval_step = partial(eval_step, callback=wandb.log, config=config)
for epoch in tqdm(range(config.epochs)):
    ## eval
    if epoch % config.log_every == 0:
        key, subkey = jr.split(key)
        eval_step(model, key=subkey)
    ## train
    for batch in dataset:
        key, subkey = jr.split(key)
        model, opt_state = train_step(model, batch, opt_state, key=subkey)
eval_step(model, key=key)

# save model
eqx.tree_serialise_leaves('model.eqx', model)
wandb.save('model.eqx')
