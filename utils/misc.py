def seed_random(seed):
    import torch

    torch.manual_seed(seed)
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def load_hydra_config(fname):
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=fname)
    return cfg


def min_bounding_num_of_power2(x):
    import math

    power = math.ceil(math.log2(x))
    return 2**power
