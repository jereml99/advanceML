import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch.distributions as td
import lightning as L
from einops import rearrange


from datamodule import FEATURE_DIM, TUDataMoudle

class GaussianPrior(torch.nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = torch.nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class SimpleGraphConv(torch.nn.Module):
    """Simple graph convolution for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        filter_length : Length of convolution filter
    """

    def __init__(self, node_feature_dim, filter_length, M = 1):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length

        # Define graph filter
        self.h = torch.nn.Parameter(1e-5 * torch.randn(filter_length))
        self.h.data[0] = 1.0

        # State output network
        self.output_net = torch.nn.Linear(self.node_feature_dim, M*2)

        self.cached = False

    def forward(self, X, A):
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
        # Compute adjacency matrices and node features per graph
 

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

   
        mean, std = torch.chunk(
            self.output_net(graph_state), 2, dim=-1
        ) 
        return td.Independent(
            td.Normal(loc=mean, scale=torch.exp(std)), 1
        )
    
class BernoulliDecoder(torch.nn.Module):
    def __init__(self, latent_dim, out_dim):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, out_dim),
        )


    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)
    

class VAE(L.LightningModule):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        
        self.save_hyperparameters()

    def elbo(self, X, A):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(X, A)
        z = q.rsample()  # Reparameterization trick
        A = rearrange(A,"b c d -> b (c d)")
        elbo = torch.mean(
            self.decoder(z).log_prob(A) - td.kl_divergence(q, self.prior()), dim=0
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return rearrange(self.decoder(z).sample(), "b (c d) -> b c d", c=28)

    def forward(self, x, edge_index, batch):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        A = to_dense_adj(edge_index, batch)
        X, idx = to_dense_batch(x, batch)
        
        return -self.elbo(X, A)
    
    def training_step(self, batch, batch_idx):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        batch: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        batch_idx: [int]
           Index of the batch.
        """
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        
        loss = self(x, edge_index, batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        
        loss = self(x, edge_index, batch)
        self.log("validation_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
        
    LATENT_DIM = 7 
    FILTER_LENGTH = 4
    datamodule = TUDataMoudle()
    
    prior = GaussianPrior(LATENT_DIM)
    encoder = SimpleGraphConv(FEATURE_DIM, FILTER_LENGTH, LATENT_DIM)
    decoder = BernoulliDecoder(LATENT_DIM, 28*28)
    VAE_model = VAE(prior, decoder, encoder)
    
    wandb_logger = L.pytorch.loggers.WandbLogger( project="GenGNN")
    trainer = L.Trainer(max_epochs=690, logger=wandb_logger)
   
    trainer.fit(VAE_model, datamodule)
    
    trainer.save_checkpoint("project3/model.ckpt")


    
    
            