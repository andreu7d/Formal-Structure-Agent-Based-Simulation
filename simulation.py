from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from TaskStructure import TaskStructure
from performance import PerformanceMetrics
from data_collection import compute_average_performance


class BlankAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.idc = 1

    def step(self):
        pass


class InteractionModel(Model):
    '''A model to model the interaction betwen agents'''

    def __init__(self, N, omega, w, alpha, beta):
        # Instantiate the model parameters
        self.num_agents, self.omega, self.w, self.alpha, self.beta = (N, omega, w, alpha, beta)

        self.task_structure = TaskStructure(N, omega, w)
        self.performance_metrics = PerformanceMetrics(N, alpha, beta)
        self.realized_matrix = self.task_structure.initial_realized_matrix()

        self.running = True
        self.schedule = RandomActivation(self)
        self.schedule.add(BlankAgent(1, self))

        self.datacollector = DataCollector(model_reporters={"AP": compute_average_performance},
                                           agent_reporters={"idc": "idc"})

    def new_project(self):
        self.task_structure.reset_task_matrix()
        self.task_structure.reset_distance_matrix()

    def update_model(self):
        social_matrix = self.task_structure.generate_social_matrix(self.realized_matrix)
        interaction_matrix = self.task_structure.generate_interaction_matrix(social_matrix)

        agent_performance = self.performance_metrics.calculate_agent_performance(self.task_structure.get_task_matrix(),
                                                                                 self.realized_matrix)

        interaction_matrix = self.performance_metrics.update_interactions_based_on_performance(interaction_matrix,
                                                                                               agent_performance)

        self.realized_matrix = self.task_structure.generate_new_realization_matrix(interaction_matrix)

    def step(self):
        self.update_model()
        self.datacollector.collect(self)
        self.schedule.step()

        # if self.schedule.steps % 50 == 0:
        #     self.new_project()







