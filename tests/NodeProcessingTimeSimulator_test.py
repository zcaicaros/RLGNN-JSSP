from semiMDP.simulators import NodeProcessingTimeSimulator
from semiMDP.utils import pprint_graph

if __name__ == "__main__":

    sim = NodeProcessingTimeSimulator(2, 2)
    g, r, done = sim.observe()

    pprint_graph(g)
