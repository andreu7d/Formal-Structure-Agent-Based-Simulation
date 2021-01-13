import numpy as np 


class TaskStructure:
    """ Task structure matrices represent the company task structures
        their theoretical optimum (Task Matrix) and
        how they are perceived by the company designers (Distance Matrix)

        Parameters
        ----------
        N : int
            The number of agents
        omega: float
            Knowledge of the designer
        w: float
            enforcement of formal structure

        Returns
        -------
        TaskStructure :obj
            The task structure of the simulation

        """

    def __init__(self, N, omega, w):
        self.num_agents = N
        self.omega = omega
        self.w = w
        self.TaskMatrix = None
        self.DistanceMatrix = None

        self.reset_task_matrix()
        self.reset_distance_matrix()

    def reset_distance_matrix(self):
        # create a matrix that shows for which interactions management got it wrong
        inverting_matrix = np.random.binomial(1, (1 - self.omega), size=(self.num_agents, self.num_agents))

        # use the matrix to flip the ones that management got wrong
        self.DistanceMatrix = np.abs(self.TaskMatrix - inverting_matrix)

        # correct the diagonals, as agents wont work with themselves
        np.fill_diagonal(self.DistanceMatrix, 0)

    def reset_task_matrix(self):
        self.TaskMatrix = np.random.binomial(1, 0.5, size=(self.num_agents, self.num_agents))

        # correct the diagonals, as agents dont have tasks with themselves
        np.fill_diagonal(self.TaskMatrix, 0)

    def initial_realized_matrix(self):
        realized = np.random.binomial(1, 0.5, size=(self.num_agents, self.num_agents))
        np.fill_diagonal(realized, 0)
        return realized

    def generate_social_matrix(self, realized_matrix):
        return self.w * self.DistanceMatrix + (1 - self.w) * realized_matrix

    def generate_interaction_matrix(self, social_matrix):
        return np.random.binomial(1, social_matrix, size=(self.num_agents, self.num_agents))

    @staticmethod
    def generate_new_realization_matrix(interaction_matrix):
        assert np.all(np.diagonal(interaction_matrix) == 0), "All diagonal elements in interaction mat should be zero"
        return interaction_matrix * interaction_matrix.T

    def get_task_matrix(self):
        return self.TaskMatrix

    def get_distance_matrix(self):
        return self.DistanceMatrix

