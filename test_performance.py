import unittest
from performance import *


class TestHelperFunctions(unittest.TestCase):

    def test_total_performance(self):
        po = PerformanceMetrics(10, 0.2, 0.3)
        perf = po.calculate_total_performance(np.ones((10, 10)), np.ones((10, 10)))
        self.assertEqual(perf, 1)
        perf = po.calculate_total_performance(np.ones((10, 10)), np.zeros((10, 10)))
        self.assertAlmostEqual(perf, 0)

    def test_performances(self):
        n = 10
        po = PerformanceMetrics(n, 0.2, 0.3)
        task_mat = np.ones((n, n))
        real_mat = np.zeros((n, n))
        np.fill_diagonal(task_mat, 0)
        np.fill_diagonal(real_mat, 0)
        agent, total = po.calculate_performances(task_mat, real_mat)
        self.assertEqual(total, 0)

        task_mat = np.ones((n, n))
        real_mat = np.ones((n, n))
        np.fill_diagonal(task_mat, 0)
        np.fill_diagonal(real_mat, 0)
        agent, total = po.calculate_performances(task_mat, real_mat)
        self.assertAlmostEqual(total, 1)

    def test_agent_performance(self):
        n = 10
        po = PerformanceMetrics(n, 0.2, 0.3)
        task_mat = np.ones((n, n))
        real_mat = np.zeros((n, n))
        np.fill_diagonal(task_mat, 0)
        np.fill_diagonal(real_mat, 0)
        perf = po.calculate_agent_performance(task_mat, real_mat)
        self.assertEqual(sum(perf), 0)

        task_mat = np.zeros((n, n))
        real_mat = np.ones((n, n))
        np.fill_diagonal(task_mat, 0)
        np.fill_diagonal(real_mat, 0)
        perf = po.calculate_agent_performance(task_mat, real_mat)
        self.assertEqual(sum(perf), 0)

        task_mat = np.ones((n, n))
        real_mat = np.ones((n, n))
        np.fill_diagonal(task_mat, 0)
        np.fill_diagonal(real_mat, 0)
        perf = po.calculate_agent_performance(task_mat, real_mat)
        self.assertEqual(sum(perf) / len(perf), 1)

        task_mat = np.zeros((n, n))
        real_mat = np.zeros((n, n))
        np.fill_diagonal(task_mat, 0)
        np.fill_diagonal(real_mat, 0)
        perf = po.calculate_agent_performance(task_mat, real_mat)
        self.assertEqual(sum(perf) / len(perf), 1)

    def test_update_interactions_based_on_performance(self):
        # Test 1
        # if none perform better than the benchmark, and the beta parameter is set to 1
        # then all should flip
        n = 10
        po = PerformanceMetrics(n, 0.5, 1)
        interaction = np.ones((n, n))
        np.fill_diagonal(interaction, 0)
        performance = np.ones(n) * 0.3
        inter2 = po.update_interactions_based_on_performance(interaction,performance)
        self.assertEqual(np.sum(inter2) , 0)

        # Test 2
        # if all perform better than expected, and the beta parameter is set to 1
        # none should flip
        interaction = np.ones((n, n))
        np.fill_diagonal(interaction, 0)
        performance = np.ones(n) * 0.7
        inter2 = po.update_interactions_based_on_performance(interaction,performance)
        self.assertEqual(np.sum(inter2)/(inter2.size - n), 1)

        # Test 3
        # if the required performance is 0.5 and half of the values will be flipped
        po = PerformanceMetrics(n, 0.5, 0.5)
        interaction = np.ones((n, n))
        np.fill_diagonal(interaction, 0)

        # and half the agents under perform, and half over perform
        performance = np.linspace(0, 1, 10)
        inter2 = po.update_interactions_based_on_performance(interaction,performance)

        # then 0.23333 % should flip, but since that is awkward, dividing by the full size should give 0.7
        self.assertEqual(np.sum(inter2)/inter2.size, 0.7)

class TestStochastic(unittest.TestCase):
    def test_totalPerformance(self):
        acc = 0.01
        prob = 0.4
        perf = PerformanceMetrics.calculate_total_performance(np.random.binomial(1, prob, size=(1000, 1000)), np.ones((1000, 1000)))
        self.assertTrue((perf - prob) < acc)

if __name__ == '__main__':
    unittest.main()
