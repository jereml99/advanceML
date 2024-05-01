from collections import defaultdict
from typing import List
import networkx as nx
import numpy as np


class Baseline:
    def __init__(self, training_samples: List[nx.graph.Graph]) -> None:
        np.random.seed(42)
        self.node_number_distribution = [
            training_sample.number_of_nodes() for training_sample in training_samples
        ]
        self.edge_number_distribution_per_number_of_nodes = defaultdict(list)
        for training_sample in training_samples:
            self.edge_number_distribution_per_number_of_nodes[
                training_sample.number_of_nodes()
            ].append(training_sample.number_of_edges())

    def sample(self, number_of_samples=1) -> List[nx.graph.Graph]:
        node_numbers = np.random.choice(
            self.node_number_distribution, number_of_samples, replace=True
        )
        samples = []

        for node_number in node_numbers:
            edge_number_distribution = (
                self.edge_number_distribution_per_number_of_nodes[node_number]
            )
            p = (sum(edge_number_distribution) / len(edge_number_distribution)) / (
                node_number * (node_number - 1) / 2
            )

            samples.append(nx.erdos_renyi_graph(node_number, p))

        return samples


if __name__ == "__main__":
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    nx.add_path(G1, [1, 2, 3, 4, 5, 6], weight=1)
    nx.add_path(G2, [10, 20, 30, 40], weight=2)
    training_samples = [G1, G2, G1, G2, G1, G2]

    b = Baseline(training_samples)
    samples = b.sample(1)

    print(list(samples[0].edges()))
