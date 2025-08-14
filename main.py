# -*- coding: utf-8 -*-

import logging

import hydra

import experiment
from experiment import ExpBase
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="main")
def main(config):
    exp: ExpBase = getattr(experiment, config.exp.name)(config)
    exp.run()


if __name__ == "__main__":
    main()