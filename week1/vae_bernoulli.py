# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.1 (2024-01-29)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(
            self.encoder_net(x), 2, dim=-1
        )  # It's for spliting the encoder output into mean and std
        return td.Independent(
            td.Normal(loc=mean, scale=torch.exp(std)), 1
        )  # It creating one multiy dimensional gaussian distribution out of X (latent space size) distribution


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
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

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()  # Reparameterization trick
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0
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
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    num_steps = len(data_loader) * epochs
    epoch = 0

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 == 0:
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step + 1) % len(data_loader) == 0:
                epoch += 1


def test(model, data_loader, device):
    """
    Test a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to test.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for testing.
    device: [torch.device]
        The device to use for testing.
    """
    model.eval()
    with torch.no_grad():
        elbo = 0
        for x, _ in data_loader:
            x = x.to(device)
            elbo += model.elbo(x).sum().item()
        elbo /= len(data_loader.dataset)
    return elbo


def plot(model, data_loader, device):
    model.eval()  # Put the model in evaluation mode

    # Prepare lists to collect all latent variables and labels
    latent_variables = []
    labels = []

    # Encode all data and store their latent variables and labels
    for (
        x,
        y,
    ) in data_loader:  # Assuming the data_loader provides the input data and labels
        x = x.to(device)
        with torch.no_grad():
            q = model.encoder(x)
            z = q.rsample()
            latent_variables.append(z)
            labels.append(y)

    # Concatenate all batch latent variables and labels into single tensors
    latent_variables = torch.cat(latent_variables, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    # Check if we need to do PCA

    if latent_variables.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_variables_2d = pca.fit_transform(latent_variables)
    else:
        latent_variables_2d = latent_variables

    # Plotting
    plt.figure(figsize=(8, 6))
    for i in np.unique(labels):
        indices = labels == i
        plt.scatter(
            latent_variables_2d[indices, 0],
            latent_variables_2d[indices, 1],
            label=f"Class {i}",
        )
    plt.title("Latent space representation colored by class label")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.savefig("latent_space.png")


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "test", "plot"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.pt",
        help="file to save model to or load model from (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: (thresshold < x).float().squeeze()),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: (thresshold < x).float().squeeze()),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M * 2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == "train":
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == "test":
        # Load model
        model.load_state_dict(
            torch.load(args.model, map_location=torch.device(args.device))
        )

        # Test model
        elbo = test(model, mnist_test_loader, args.device)
        print(f"ELBO: {elbo:.1f}")

    elif args.mode == "sample":
        model.load_state_dict(
            torch.load(args.model, map_location=torch.device(args.device))
        )

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)
    elif args.mode == "plot":
        model.load_state_dict(
            torch.load(args.model, map_location=torch.device(args.device))
        )
        plot(model, mnist_test_loader, args.device)
