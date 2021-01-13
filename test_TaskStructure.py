import unittest
import numpy as np
from TaskStructure import TaskStructure

class TestTaskStructure(unittest.TestCase):

    def test_instatiate(self):
        ts = TaskStructure(100, 1, 0.5)
        self.assertTrue(np.all(np.unique(ts.get_task_matrix()) == np.array([0, 1])))

    def test_DistanceMatrix(self):
        ts = TaskStructure(5, 1, 0.5)
        self.assertTrue(np.all(ts.get_task_matrix() == ts.get_distance_matrix()))

        ts = TaskStructure(5, 0, 0.5)
        self.assertEqual(np.sum(ts.get_task_matrix() * ts.get_distance_matrix()), 0)

    def test_SocialMatrix(self):
        ts = TaskStructure(5, 0.5, 0)
        init_RM = ts.initial_realized_matrix()
        self.assertTrue(np.all(ts.generate_social_matrix(init_RM) == init_RM))

        ts = TaskStructure(5, 0.5, 1)
        init_RM = ts.initial_realized_matrix()
        self.assertTrue(np.all(ts.generate_social_matrix(init_RM) == ts.get_distance_matrix()))

    def test_InteractionMatrix(self):
        ts = TaskStructure(5, 0.5, 1)
        mat = np.zeros((5, 5))
        self.assertTrue(np.all(ts.generate_interaction_matrix(mat) == mat))
        mat = np.ones((5, 5))
        self.assertTrue(np.all(ts.generate_interaction_matrix(mat) == mat))

    def test_generate_new_realized_matrix(self):
        n = 5
        ts = TaskStructure(n, 0.5, 0.5)
        mat = np.ones((n, n))
        np.fill_diagonal(mat, 0)
        realized = ts.generate_new_realization_matrix(mat)
        self.assertTrue(np.sum(realized) == (n*(n-1)))

        mat[[1, 1, 3, 4], [2, 3, 2, 3]] = 0
        realized = ts.generate_new_realization_matrix(mat)
        test_result = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]]
        self.assertTrue(np.all(realized == test_result))



class TestStochastic(unittest.TestCase):

    def test_InteractionMatrixStoch(self):
        N_agent = 4
        n_round = 10000
        acc = 0.01

        ts = TaskStructure(N_agent, 0.4, 0.5)
        total = np.zeros((N_agent, N_agent))

        realized_mat = ts.initial_realized_matrix()
        social_mat = ts.generate_social_matrix(realized_mat)

        # generate n_rounds of Interaction matrices
        # sum them up and later we will divide it by the total rounds
        # since its a Bernoully dist, it wil also give us the probability
        for i in range(n_round):
            total += ts.generate_interaction_matrix(social_mat)

        # check that it really creates the correct distribution
        # by checking the differences between the generated prop
        # and the input prob
        diff = (total / n_round) - social_mat
        np.all(diff < acc)


if __name__ == '__main__':
    unittest.main()
