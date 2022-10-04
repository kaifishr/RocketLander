"""Optimizer class for landing a booster.
"""
import numpy
import random

numpy.random.seed(0)
random.seed(0)

from torch.utils.tensorboard import SummaryWriter

from environment import Environment
from config import load_config, Config


class Optimizer: 
    """Contains logic for optimization.

    Optimizer uses Environment to interact with physics simulation.
    Environment acts as interface between Optimizer and Booster.

    Attributes:
        env:
        booster:
        data:
    
    """
    def __init__(self, config: Config):

        # Environemnt creates the interface between booster and optimizer.
        self.env = Environment(config=config)
        self.boosters = self.env.boosters

        self.data = None
        self.fitness = 0.0

        self.writer = SummaryWriter()

    def comp_action(self):
        """Computes action based on current data.

        This module holds the boosters brain.
        
        """
        for booster, data in zip(self.boosters, self.data):
            print(booster.control_system(data))

    def optimize(self):
        """
        'optimize' calls Framework's 'run' method for rendering and
        physics step performed by 'Step'. Currently, only access to
        simulation data and object manipulation is possible within
        the 'Step' as 'run' method is a while-loop.
        """
        iteration = 0

        while True:

            score = 0.0
            self.writer.add_scalar("training/score", score, iteration)

            iteration += 1


        # Simulation step
        # while True: 
        #     self.env.run_()        # Minimal Framework, this works with while-loop

        self.env.run()          # Standard Framework

        # Read / extract booster data.
        # self.data = self.env.read_data()

        # Process booster data and compute actions.
        # self.comp_action()

        # if self.iteration % self.max_simulation_steps == 0:
        #     self.env.reset()


if __name__ == "__main__":

    config = load_config(path="../config.yml")
    print(config)
    optimizer = Optimizer(config=config)
    optimizer.optimize()
