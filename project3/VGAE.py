import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch.distributions as td
import lightning as L
from einops import rearrange
from torch.nn import functional as F

from datamodule import FEATURE_DIM, TUDataMoudle
from torch.optim.lr_scheduler import StepLR

class GaussianPriorVGAE(torch.nn.Module):
    def __init__(self, latent_dim, node_count=28):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPriorVGAE, self).__init__()
        self.latent_dim = latent_dim
        self.mean = torch.nn.Parameter(torch.zeros((node_count,self.latent_dim)), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones((node_count,self.latent_dim)), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 2)

class InnerProductDecoder(torch.nn.Module):
    def __init__(self):
        """
        Define an inner product decoder distribution.
        """
        super(InnerProductDecoder, self).__init__()

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        A_pred = torch.bmm(z, z.permute(0, 2, 1)) # .permute(0, 2, 1) is T
        return td.Independent(td.Bernoulli(logits=torch.sigmoid(A_pred)), 2)

class VGAE(L.LightningModule):
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

        super(VGAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

        self.save_hyperparameters()

    def elbo(self, X, A):
        """
        Compute the ELBO for the given batch of data.
        """

        q = self.encoder(X, A)
        z = q.rsample()  # Reparameterization trick
        entropy = self.decoder(z).log_prob(A)
        elbo = torch.mean(entropy - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x, edge_index, batch):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        A = to_dense_adj(edge_index, batch, max_num_nodes=28)
        X, idx = to_dense_batch(x, batch, max_num_nodes=28)
        elbo = -self.elbo(X, A)
        return elbo
    

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = {
            "scheduler": StepLR(optimizer, step_size=10, gamma=0.9),
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

class GraphConvEncoder(torch.nn.Module):
    def __init__(self, node_feature_dim, filter_length, M):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length

        # Define graph filter
        self.gcn_mean = SimpleGraphConv(node_feature_dim, filter_length, M)
        self.gcn_std = SimpleGraphConv(node_feature_dim, filter_length, M)

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
        mean = self.gcn_mean(X, A)
        std = self.gcn_std(X, A)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 2)


class SimpleGraphConvVGAE(torch.nn.Module):
    """Simple graph convolution for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        filter_length : Length of convolution filter
    """

    def __init__(self, node_feature_dim, filter_length, M=1):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length

        # Define graph filter
        self.h = torch.nn.Parameter(1e-5 * torch.randn(filter_length), requires_grad=True)
        self.h.data[0] = 1.0

        # State output network
        self.output_net = torch.nn.Linear(self.node_feature_dim, M )

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
        L, U = torch.linalg.eigh(A)        
        exponentiated_L = L.unsqueeze(2).pow(torch.arange(self.filter_length, device=L.device))
        diagonal_filter = (self.h[None,None] * exponentiated_L).sum(2, keepdim=True)
        node_state = U @ (diagonal_filter * (U.transpose(1, 2) @ X))
        output = self.output_net(node_state)
        return output

if __name__ == "__main__":
    LATENT_DIM = 8
    FILTER_LENGTH = 4
    datamodule = TUDataMoudle()

    prior = GaussianPrior(LATENT_DIM)
    encoder = GraphConvEncoder(FEATURE_DIM, FILTER_LENGTH, LATENT_DIM)
    decoder = InnerProductDecoder()
    VAE_model = VGAE(prior, decoder, encoder)
    VAE_model.sample()
    wandb_logger = L.pytorch.loggers.WandbLogger(project="GenGNN")
    trainer = L.Trainer(
        max_epochs=1000,
        logger=wandb_logger,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="train_loss",
                dirpath="project3",
                filename="model-{epoch:02d}-{train_loss:.2f}",
            ),
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="validation_loss",
                dirpath="project3",
                filename="model-{epoch:02d}-{validation_loss:.2f}",
            ),
        ],
    )

    trainer.fit(VAE_model, datamodule)

    trainer.save_checkpoint("project3/model.ckpt")