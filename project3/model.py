import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch


class SimpleGraphConv(torch.nn.Module):
    """Simple graph convolution for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        filter_length : Length of convolution filter
    """

    def __init__(self, node_feature_dim, filter_length):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length

        # Define graph filter
        self.h = torch.nn.Parameter(1e-5 * torch.randn(filter_length))
        self.h.data[0] = 1.0

        # State output network
        self.output_net = torch.nn.Linear(self.node_feature_dim, 1)

        self.cached = False

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max() + 1

        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(edge_index, batch)
        X, idx = to_dense_batch(x, batch)

        # ---------------------------------------------------------------------------------------------------------

        big_lambda, U = torch.linalg.eigh(A)
        big_lambda = torch.diag_embed(big_lambda)
        U_star = U.transpose(-2, -1).conj()

        lambda_sum = torch.zeros_like(big_lambda)
        for k in range(self.filter_length):
            lambda_sum += self.h[k] * torch.linalg.matrix_power(big_lambda, k)
        node_state = U @ lambda_sum @ U_star @ X

        # Aggregate the node states
        graph_state = node_state.sum(1)

        # Output
        out = self.output_net(graph_state).flatten()
        return out
