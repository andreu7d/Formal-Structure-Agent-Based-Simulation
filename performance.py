import numpy as np


class PerformanceMetrics:

    def __init__(self, N, alpha, beta):
        self.N = N
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def calculate_agent_performance(task_matrix: np.array, realized_matrix: np.array) -> np.array:
        """
        Calculates the performance of each agent based on the task and realized structure matrices
        Parameters
        ----------
        task_matrix : NxN binary array
            matrix representing the ideal task structure
        realized_matrix : NxN binary array
            matrix representing the realized interaction structure

        Raises
        ------
        AssertException
            If the input matrices diagonals are not zero. Since each agent should not interact with himself

        Returns
        -------
        agent_performance: array float  1XN
            performance of each agent in [0,1]

        """
        # check that the diagonals of the matrices are zero
        assert np.all((np.diagonal(task_matrix) + np.diagonal(realized_matrix)) == 0), \
            "Diagonals of matrices are not zero"

        # the maximum wrong interactions can be the total minus itself
        max_wrong = task_matrix.shape[0] - 1

        # calculate the performance per agent
        return 1 - (np.sum(abs(task_matrix - realized_matrix), axis=1) / max_wrong)

    @staticmethod
    def calculate_total_performance(task_matrix, realized_matrix):
        return 1 - (np.sum(abs((task_matrix - realized_matrix)) / task_matrix.size))

    def calculate_performances(self, task_matrix, realized_matrix):
        agent_perf = self.calculate_agent_performance(task_matrix, realized_matrix)
        total_perf = sum(agent_perf) / len(agent_perf)

        return agent_perf, total_perf

    def update_interactions_based_on_performance(self, interaction_matrix: np.array, agent_performance: np.array) \
            -> np.ndarray:
        """
        if the performance of an agent drops below alpha (the desired level) then a proportion beta of their interactions
        in the performance matrix will flip. This function calculates this.

        Parameters
        ----------
        interaction_matrix: array
            the previously generated interaction matrix

        agent_performance: array
            the agent performance in the previous time step

        Returns
        -------
        interaction_matrix: array
            recalculated interaction array

        """
        # if beta is 0 do nothing, no need to go through the for loop
        if self.beta == 0:
            return interaction_matrix

        # assert that the diagonals are 0
        assert np.all(np.diagonal(interaction_matrix) == 0)

        inverting_mat = np.zeros_like(interaction_matrix)
        prop = int(self.beta * (self.N - 1))  # N-1 because it should not change itself

        for i in range(self.N):
            if agent_performance[i] < self.alpha:
                choice_list = np.concatenate((np.arange(0, i), np.arange(i+1, self.N)))  # avoid flipping its own
                # Choose from the list of indexes which one to flip randomly
                flip_idx = np.random.choice(choice_list, size=prop, replace=False)
                inverting_mat[i, flip_idx] = 1

        # Use the matrix that determines which ones to invert, to invert the interaction matrix
        return abs(inverting_mat - interaction_matrix)
