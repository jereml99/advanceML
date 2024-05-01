import networkx as nx
from networkx import weisfeiler_lehman_graph_hash


def novel(generated_samples, training_samples):
    return sum(
        [
            all(
                [
                    not (weisfeiler_lehman_graph_hash(generated_sample) == weisfeiler_lehman_graph_hash(training_sample))
                    for training_sample in training_samples
                ]
            )
            for generated_sample in generated_samples
        ]
    ) / len(generated_samples)


def unique(generated_samples):
    unique_samples = []
    for generated_sample in generated_samples:
        if (
            all(
                [
                    not (weisfeiler_lehman_graph_hash(generated_sample) == weisfeiler_lehman_graph_hash(unique_sample))
                    for unique_sample in unique_samples
                ]
            )
            or len(unique_samples) == 0
        ):
            unique_samples.append(generated_sample)

    return len(unique_samples) / len(generated_samples)


def degree_histogram(samples):
    result = []
    for sample in samples:
        result.extend(list(dict(nx.degree(sample)).values()))
    return result


def clustering_coefficient_histogram(samples):
    clustering_coefficients = []
    for sample in samples:
        clustering_coefficients.extend(list(nx.clustering(sample).values()))
    return clustering_coefficients


def eigenvector_centrality_histogram(samples):
    eigenvector_centralitys = []
    for sample in samples:
        eigenvector_centralitys.extend(
            list(nx.eigenvector_centrality(sample, max_iter=10000).values())
        )
    return eigenvector_centralitys


if __name__ == "__main__":
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    nx.add_path(G1, [1, 2, 3, 4, 5, 6], weight=1)
    nx.add_path(G2, [10, 20, 30, 40], weight=2)
    Generated_samples = [G1, G2, G1, G2, G1, G2]

    G3 = nx.DiGraph()
    G4 = nx.DiGraph()
    nx.add_path(G3, [1, 2, 3, 4, 5], weight=1)
    nx.add_path(G4, [10, 20, 30, 40], weight=2)
    Training_samples = [G3, G4, G3, G4, G1, G2]
    print(f"Novel: {novel(Generated_samples, Training_samples)}")

    print(f"Unique Generated: {unique(Generated_samples)}")
    print(f"Unique training: {unique(Training_samples)}")

    print(f"Degree histogram: {degree_histogram(Generated_samples)}")
    print(
        f"Clustering coefficient histogram: {clustering_coefficient_histogram(Generated_samples)}"
    )
    print(
        f"Eigenvector centrality histogram: {eigenvector_centrality_histogram(Generated_samples)}"
    )
