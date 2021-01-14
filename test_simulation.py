import unittest
from simulation import InteractionModel

class SimulationIntegration(unittest.TestCase):

    def test_one_step(self):
        sim = InteractionModel(N=15, omega=0.5, w=0.2, alpha=0.4, beta=0.1)
        sim.step()



if __name__ == '__main__':
    unittest.main()
