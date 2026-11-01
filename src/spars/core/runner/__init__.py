"""Experiment execution; import the optional RL runtime only when requested."""


def run_with_config(cfg):
    from .run import run_with_config as execute

    return execute(cfg)
