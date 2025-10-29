from functools import partial

import jax.random as jr
import tyro
import wandb
from tqdm.auto import tqdm

from config import default_configs
from utils import eval_step, make_dataset, make_model, make_opt, save_model, train_step

# configuration
config = tyro.extras.overridable_config_cli(default_configs)
key = jr.key(config.seed)
wandb.init(project='ngem', name=repr(config), config=config.asdict())

# init dataset
key, key1, key2 = jr.split(key, 3)
train_set = make_dataset(config, key=key1, train=True)
eval_set = make_dataset(config, key=key2, train=False)

# init mode
key, subkey = jr.split(key)
model = make_model(config, key=subkey)
opt, opt_state = make_opt(config, model)

# main loop
train_step = partial(train_step, callback=wandb.log, opt=opt)
eval_step = partial(eval_step, callback=wandb.log, config=config, eval_set=eval_set)
for epoch in tqdm(range(config.epochs)):
    ## eval
    if epoch % config.log_every == 0:
        key, subkey = jr.split(key)
        eval_step(model, key=subkey)
    ## train
    for batch in train_set:
        key, subkey = jr.split(key)
        model, opt_state = train_step(model, batch, opt_state, key=subkey)
eval_step(model, key=key)
save_model(model)
