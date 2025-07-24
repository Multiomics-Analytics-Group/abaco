import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import math
from abaco.BatchEffectDataLoader import class_to_int, one_hot_encoding
import random

# ---------- PRIOR CLASSES DEFINITIONS ---------- #


class NormalPrior(nn.Module):
    def __init__(self, d_z):
        """
        Define a Gaussian Normal prior distribution with zero mean and unit variance (Standard distribution).

        Parameters:
            d_z: [int]
                Dimension of the latent space
        """
        super().__init__()
        self.d_z = d_z
        self.mu = nn.Parameter(torch.zeros(self.d_z), requires_grad=False)
        self.var = nn.Parameter(torch.ones(self.d_z), requires_grad=False)

    def forward(self):
        """
        Return prior distribution. This allows the computation of KL-divergence by calling self.prior() in the VAE class.

        Returns:
            prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mu, scale=self.var), 1)


class MoGPrior(nn.Module):
    def __init__(self, d_z, n_comp, multiplier=1.0):
        """
        Define a Mixture of Gaussian Normal prior distribution.

        Parameters:
            d_z: [int]
                Dimension of the latent space
            n_comp: [int]
                Number of components for the MoG distribution
            multiplier: [float]
                Parameter that controls sparsity of each Gaussian component
        """
        super().__init__()
        self.d_z = d_z
        self.n_comp = n_comp

        self.mu = nn.Parameter(torch.randn(n_comp, self.d_z) * multiplier)
        self.var = nn.Parameter(torch.randn(n_comp, self.d_z))
        self.pi = nn.Parameter(torch.zeros(n_comp))

    def forward(self):
        """
        Return prior distribution, allowing for the computation of the KL-divergence by calling self.prior().

        Returns:
            prior: [torch.distributions.Distribution]
        """
        # Get parameters for each MoG component
        means = self.mu
        stds = torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)
        logits = self.pi

        # Call MoG distribution
        prior = MixtureOfGaussians(logits, means, stds)

        return prior


class VMMPrior(nn.Module):
    def __init__(self, d_z, n_comp, mu, multiplier=1.0):
        """
        Define a Mixture of Gaussian Normal prior distribution.

        Parameters:
            d_z: [int]
                Dimension of the latent space
            n_comp: [int]
                Number of components for the MoG distribution
            multiplier: [float]
                Parameter that controls sparsity of each Gaussian component
        """
        super().__init__()
        self.d_z = d_z
        self.n_comp = n_comp

        self.mu = mu
        self.var = nn.Parameter(torch.randn(n_comp, self.d_z))
        self.pi = nn.Parameter(torch.zeros(n_comp))

    def forward(self):
        """
        Return prior distribution, allowing for the computation of the KL-divergence by calling self.prior().

        Returns:
            prior: [torch.distributions.Distribution]
        """
        # Get parameters for each MoG component
        means = self.mu
        stds = torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)
        logits = self.pi

        # Call MoG distribution
        prior = MixtureOfGaussians(logits, means, stds)

        return prior


# ---------- ENCODER CLASSES DEFINITIONS ---------- #


class NormalEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian Normal encoder to obtain the parameters of the Normal distribution.

        Parameters:
            encoder_net: [torch.nn.Module]
                The encoder network, takes a tensor of dimension (batch, features) and
                outputs a tensor of dimension (batch, 2*d_z), where d_z is the dimension of the
                latent space.
        """
        super().__init__()
        self.encoder_net = encoder_net

    def encode(self, x):

        mu, var = torch.chunk(
            self.encoder_net(x), 2, dim=-1
        )  # chunk is used for separating the encoder output (batch, 2*d_z) into two separate vectors (batch, d_z)

        return mu, var

    def forward(self, x):
        """
        Computes the Gaussian Normal distribution over the latent space.

        Parameters:
            x: [torch.Tensor]
        """
        mu, var = self.encode(x)
        std = torch.sqrt(torch.nn.functional.softplus(var) + 1e-8)

        return td.Independent(td.Normal(loc=mu, scale=std), 1)

    def det_encode(self, x):
        """
        Computes the encoded point without stochastic component.

        Parameters:
            x: torch.Tensor
        """
        mu, _ = self.encode(x)

        return mu

    def monte_carlo_encode(self, x, K=100):
        mu, var = self.encode(x)
        std = torch.sqrt(torch.nn.functional.softplus(var) + 1e-8)

        dist = td.Independent(td.Normal(loc=mu, scale=std), 1)

        samples = dist.sample(sample_shape=(K,))

        return samples.mean(dim=0)


class MoGEncoder(nn.Module):
    def __init__(self, encoder_net, n_comp):
        """
        Define a Mixture of Gaussians encoder to obtain the parameters of the MoG distribution.

        Parameters:
            encoder_net: [torch.nn.Module]
                The encoder network, takes a tensor of dimension (batch, features) and
                outputs a tensor of dimension (batch, n_comp*(2*d_z + 1)), where d_z is the dimension
                of the latent space, and n_comp the number of components of the MoG distribution.
            n_comp: [int]
                Number of components for the MoG distribution.
        """
        super().__init__()
        self.n_comp = n_comp
        self.encoder_net = encoder_net

    def encode(self, x):

        comps = torch.chunk(
            self.encoder_net(x), self.n_comp, dim=-1
        )  # chunk used for separating the encoder output (batch, n_comp*(2*d_z + 1)) into n_comp separate vectors (batch, n_comp)

        # Parameters list (for extracting in loop)
        mu_list = []
        var_list = []
        pi_list = []

        for comp in comps:
            params = comp[
                :, :-1
            ]  # parameters mu and var are on the 2*d_z first values of the component
            pi_comp = comp[
                :, -1
            ]  # mixing probabilities is the last value of the component

            mu, var = torch.chunk(
                params, 2, dim=-1
            )  # separating mu from var using chunk

            mu_list.append(mu)
            var_list.append(var)
            pi_list.append(pi_comp)

        # Convert parameters list into tensor
        means = torch.stack(mu_list, dim=1)
        stds = torch.sqrt(
            torch.nn.functional.softplus(torch.stack(var_list, dim=1)) + 1e-8
        )
        pis = torch.stack(pi_list, dim=1)

        # Clamp to avoid error values (too low or too high)
        stds = torch.clamp(stds, min=1e-5, max=1e5)

        return pis, means, stds

    def forward(self, x):
        """
        Computes the MoG distribution over the latent space.

        Parameters:
            x: [torch.Tensor]
        """
        comps = torch.chunk(
            self.encoder_net(x), self.n_comp, dim=-1
        )  # chunk used for separating the encoder output (batch, n_comp*(2*d_z + 1)) into n_comp separate vectors (batch, n_comp)

        # Parameters list (for extracting in loop)
        mu_list = []
        var_list = []
        pi_list = []

        for comp in comps:
            params = comp[
                :, :-1
            ]  # parameters mu and var are on the 2*d_z first values of the component
            pi_comp = comp[
                :, -1
            ]  # mixing probabilities is the last value of the component

            mu, var = torch.chunk(
                params, 2, dim=-1
            )  # separating mu from var using chunk

            mu_list.append(mu)
            var_list.append(var)
            pi_list.append(pi_comp)

        # Convert parameters list into tensor
        means = torch.stack(mu_list, dim=1)
        stds = torch.sqrt(
            torch.nn.functional.softplus(torch.stack(var_list, dim=1)) + 1e-8
        )
        pis = torch.stack(pi_list, dim=1)

        # Clamp to avoid error values (too low or too high)
        stds = torch.clamp(stds, min=1e-5, max=1e5)
        # Create individual Gaussian distribution per component
        mog_dist = MixtureOfGaussians(pis, means, stds)

        return mog_dist

    def det_encode(self, x):
        """
        Computes the encoded point without stochastic component.

        Parameters:
            x: torch.Tensor
        """
        comps = torch.chunk(
            self.encoder_net(x), self.n_comp, dim=-1
        )  # chunk used for separating the encoder output (batch, n_comp*(2*d_z + 1)) into n_comp separate vectors (batch, n_comp)

        # Parameters list (for extracting in loop)
        mu_list = []
        pi_list = []

        for comp in comps:
            params = comp[
                :, :-1
            ]  # parameters mu and var are on the 2*d_z first values of the component
            pi_comp = comp[
                :, -1
            ]  # mixing probabilities is the last value of the component

            mu, _ = torch.chunk(params, 2, dim=-1)  # separating mu from var using chunk

            mu_list.append(mu)
            pi_list.append(pi_comp)

        # Convert parameters list into tensor
        means = torch.stack(mu_list, dim=1)
        pis = F.softmax(torch.stack(pi_list, dim=1))

        z = torch.einsum("bn,bnd->bd", pis, means)

        return z

    def monte_carlo_encode(self, x, K=100):
        """
        Computes the encoded point with stochastic component.

        Parameters:
            x: torch.Tensor
        """
        comps = torch.chunk(
            self.encoder_net(x), self.n_comp, dim=-1
        )  # chunk used for separating the encoder output (batch, n_comp*(2*d_z + 1)) into n_comp separate vectors (batch, n_comp)

        # Parameters list (for extracting in loop)
        mu_list = []
        var_list = []
        pi_list = []

        for comp in comps:
            params = comp[
                :, :-1
            ]  # parameters mu and var are on the 2*d_z first values of the component
            pi_comp = comp[
                :, -1
            ]  # mixing probabilities is the last value of the component

            mu, var = torch.chunk(
                params, 2, dim=-1
            )  # separating mu from var using chunk

            mu_list.append(mu)
            var_list.append(var)
            pi_list.append(pi_comp)

        # Convert parameters list into tensor
        means = torch.stack(mu_list, dim=1)
        stds = torch.sqrt(
            torch.nn.functional.softplus(torch.stack(var_list, dim=1)) + 1e-8
        )
        pis = torch.stack(pi_list, dim=1)

        # Clamp to avoid error values (too low or too high)
        stds = torch.clamp(stds, min=1e-5, max=1e5)
        # Create individual Gaussian distribution per component
        mog_dist = MixtureOfGaussians(pis, means, stds)

        samples = mog_dist.sample(sample_shape=(K,))

        return samples.mean(dim=0)


# ---------- DECODER CLASSES DEFINITIONS ---------- #


class NegativeBinomialDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Negative Binomial decoder to obtain the parameters of the Negative Binomial distribution.

        Parameters:
            decoder_net: [torch.nn.Module]
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs
                a tensor of dimension (batch, 2*features), where d_z is the dimension of the
                latent space.
        """
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Computes the Negative Binomial distribution over the data space. What we are getting is the mean
        and the dispersion parameters, so it is needed a parameterization in order to get the NB
        distribution parameters: total_count (dispersion) and probs (dispersion/(dispersion + mean))

        Parameters:
            z: [torch.Tensor]
        """
        mu, theta = torch.chunk(self.decoder_net(z), 2, dim=-1)
        # Ensure mean and dispersion are positive numbers
        mu = F.softplus(mu)
        theta = F.softplus(theta)
        # Parameterization into NB parameters
        p = theta / (theta + mu)
        r = theta

        return td.Independent(td.NegativeBinomial(total_count=r, probs=p), 1)


class ZINBDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Zero-inflated Negative Binomial decoder to obtain the parameters of the ZINB distribution.

        Parameters:
            decoder_net: [torch.nn.Module]
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs
                a tensor of dimension (batch, 3*features), where d_z is the dimension of the
                latent space.
        """
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Computes the Zero-inflated Negative Binomial distribution over the data space. What we are getting is the mean
        and the dispersion parameters, so it is needed a parameterization in order to get the NB
        distribution parameters: total_count (dispersion) and probs (dispersion/(dispersion + mean)). We are also
        getting the pi_logits parameters, which accounts for the zero inflation probability.

        Parameters:
            z: [torch.Tensor]
        """
        mu, theta, pi_logits = torch.chunk(self.decoder_net(z), 3, dim=-1)
        # Ensure mean and dispersion are positive numbers and pi is in range [0,1]

        mu = F.softplus(mu)
        theta = F.softplus(theta) + 1e-4

        pi = torch.sigmoid(pi_logits)

        # Parameterization into NB parameters
        p = theta / (theta + mu)

        r = theta

        # Clamp values to avoid huge / small probabilities
        p = torch.clamp(p, min=1e-5, max=1 - 1e-5)

        # Create Negative Binomial component
        nb = td.NegativeBinomial(total_count=r, probs=p)
        # nb = td.Independent(nb, 1)

        return td.Independent(ZINB(nb, pi), 1)

    def monte_carlo_decode(self, z, K=100):
        """
        Computes the Zero-inflated Negative Binomial mode (or expected value) through Monte Carlo approximation.
        Parameters:
            z: [torch.Tensor]
                Latent space point.
            K: [int]
                Number of Monte Carlo iterations.
        """
        mu, theta, pi_logits = torch.chunk(self.decoder_net(z), 3, dim=-1)
        # Ensure mean and dispersion are positive numbers and pi is in range [0,1]

        mu = F.softplus(mu)
        theta = F.softplus(theta) + 1e-4

        pi = torch.sigmoid(pi_logits)

        # Parameterization into NB parameters
        p = theta / (theta + mu)

        r = theta

        # Clamp values to avoid huge / small probabilities
        p = torch.clamp(p, min=1e-5, max=1 - 1e-5)

        # Create Negative Binomial component
        nb = td.NegativeBinomial(total_count=r, probs=p)

        # Define Zero-inflated model
        zinb = td.Independent(ZINB(nb, pi), 1)

        # Sample sample sample!
        samples = zinb.sample(sample_shape=(K,))

        return samples.mean(dim=0).floor().int()


class DirichletDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Dirichlet decoder to obtain relative abundance values of the Dirichlet distribution.

        Parameters:
            decoder_net: [torch.nn.Module]
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs
                a tensor of dimension (batch, features), where d_z is the dimension of the
                latent space.
        """
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Computes the Dirichlet distribution over the data space. What we are getting is the concentration parameter
        that is going to be the input for the td.Dirichlet distribution function.

        Parameters:
            z: [torch.Tensor]
        """
        concentration = self.decoder_net(z)
        # For stability, value is required to be positive and greater than 0
        concentration = F.softplus(concentration + 1e-4)
        dirichlet_dist = td.Dirichlet(concentration)
        return td.Independent(dirichlet_dist, 1)


class ZIDirichletDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Zero-Inflated Dirichlet decoder to obtain relative abundance values of the ZIDirichlet distribution.

        Parameters:
            decoder_net: [torch.nn.Module]
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs
                a tensor of dimension (batch, 2*features), where d_z is the dimension of the
                latent space.
        """
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Computes the Zero-Inflated Dirichlet distribution over the data space. What we are getting is the concentration
        parameter that is going to be the input for the ZIDirichlet distribution function.

        Parameters:
            z: [torch.Tensor]
        """
        concentration, pi_logits = torch.chunk(self.decoder_net(z), 2, dim=-1)

        concentration = F.softplus(concentration) + 1e-4

        pi = torch.sigmoid(pi_logits)

        dirichlet_dist = td.Dirichlet(concentration)
        return td.Independent(ZIDirichlet(pi, dirichlet_dist), 1)


# ---------- DISTRIBUTIONS CLASSES DEFINITIONS ---------- #


class ZINB(td.Distribution):
    """
    Zero-inflated Negative Binomial (ZINB) distribution definition. The ZINB distribution is defined by the following:
        For x = 0:
            ZINB.log_prob(0) = log(pi + (1 - pi) * NegativeBinomial(0 | r, p))

        For x > 0:
            ZINB.log_prob(x) = log(1 - pi) + NegativeBinomial(x |r, p).log_prob(x)
    """

    arg_constraints = {}
    support = td.constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, nb, pi, validate_args=None):
        """
        Parameters:
            nb: [torch.distributions.Independent(torch.distributions.NegativeBinomial)]
                Negative Binomial distribution component
            pi: [torch.Tensor]
                Zero-inflation probability from decoder output
        """
        self.nb = nb
        self.pi = pi
        batch_shape = nb.batch_shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, x):
        """
        Defines the log_prob() function inherent from torch.distributions.Distribution.

        Parameters:
            x: [torch.Tensor]
        """
        nb_log_prob = self.nb.log_prob(x)  # log probability of NB where x > 0
        nb_prob_zero = torch.exp(
            self.nb.log_prob(torch.zeros_like(x))
        )  # probability of NB where x = 0
        log_prob_zero = torch.log(self.pi + (1 - self.pi) * nb_prob_zero + 1e-8)
        log_prob_nonzero = torch.log(1 - self.pi + 1e-8) + nb_log_prob

        return torch.where(x == 0, log_prob_zero, log_prob_nonzero)

    def sample(self, sample_shape=torch.Size()):
        """
        Defines the sample() function, which is used to sample data points using the distribution parameters.
            For x = 0:
                Binary mask for zero-inflated values
            For x > 0:
                Sample from Negative Binomial distribution
        """
        shape = self._extended_shape(sample_shape)
        zero_inflated = torch.bernoulli(self.pi.expand(shape))

        nb_sample = self.nb.sample(sample_shape)

        return torch.where(zero_inflated.bool(), torch.zeros_like(nb_sample), nb_sample)


@td.kl.register_kl(ZINB, ZINB)
def kl_zinb_zinb(p, q):
    # Monte Carlo sampling from p
    num_samples = 5000
    samples = p.sample((num_samples,))
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)

    kl = (log_p - log_q).mean(dim=0)
    return kl


class ZIDirichlet(td.Distribution):
    """
    A Zero-inflated Dirichlet distribution.
    """

    arg_constraints = {
        "zero_prob": td.constraints.unit_interval,
    }
    support = td.constraints.simplex
    has_rsample = False

    def __init__(self, zero_prob, dirichlet, validate_args=None):
        self.zero_prob = zero_prob
        self.dirichlet = dirichlet
        batch_shape = dirichlet.batch_shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, x):
        """
        Defines the log_prob() function inherent from torch.distributions.Distribution.

        Parameters:
            x: [torch.Tensor]
        """
        # Sanity check, if all elements in the last dimension are zero
        is_zero = (x == 0).all(dim=-1)

        # Compute Dirichlet log probability for all sampels
        dirichlet_log_prob = self.dirichlet.log_prob(x)
        print(dirichlet_log_prob)

        log_prob_zero = torch.log(self.zero_prob + 1e-8)
        log_prob_nonzero = torch.log(1 - self.zero_prob + 1e-8) + dirichlet_log_prob

        return torch.where(is_zero, log_prob_zero, log_prob_nonzero)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        zero_inflated = torch.bernoulli(self.zero_prob.expand(shape))

        dirichlet_sample = self.dirichlet.sample(sample_shape)

        return torch.where(
            zero_inflated.bool(), torch.zeros_like(dirichlet_sample), dirichlet_sample
        )


@td.kl.register_kl(ZIDirichlet, ZIDirichlet)
def kl_zidirichlet_zidirichlet(p, q):
    # Monte Carlo sampling from p
    num_samples = 5000
    samples = p.sample((num_samples,))
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)

    kl = (log_p - log_q).mean(dim=0)
    return kl


class MixtureOfGaussians(td.Distribution):
    """
    A Mixture of Gaussians distribution with reparameterized sampling. Computation of gradients is possible.
    """

    arg_constraints = {}
    support = td.constraints.real
    has_rsample = True  # Implemented the rsample() through the Gumbel-softmax reparameterization trick.

    def __init__(
        self, mixture_logits, means, stds, temperature=1e-5, validate_args=None
    ):
        self.mixture_logits = mixture_logits
        self.means = means
        self.stds = stds
        self.temperature = temperature

        batch_shape = self.mixture_logits.shape[:-1]
        event_shape = self.means.shape[-1:]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampling using the Gubel-softmax trick.
        """

        # Step 1 - Sample for every component

        logits = self.mixture_logits.expand(sample_shape + self.mixture_logits.shape)
        means = self.means.expand(sample_shape + self.means.shape)
        stds = self.stds.expand(sample_shape + self.stds.shape)

        eps = torch.randn_like(means)
        comp_samples = means + eps * stds

        # Step 2 - Generate Gumbel noise for each component
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-5) + 1e-5)

        # Step 3 - Compute y_i (gumbel-softmax trick)
        weights = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        weights = weights.unsqueeze(-1)

        # Step 4 - Sum every component for final sampling
        sample = torch.sum(weights * comp_samples, dim=-2)
        return sample

    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the MoG distribution.
        """
        return self.rsample(sample_shape)

    def log_prob(self, value):
        """
        Compute the log probability of a given value. The log prob of a MoG is defined as:

            log_prob(x) = log [sum_k (pi_k * N(x; mu_k, sigma_k^2)]

        Where pi_k are the mixture probabilities.
        """
        value = value.unsqueeze(-2)

        normal = td.Normal(self.means, self.stds)
        log_prob_comp = normal.log_prob(value)
        log_prob_comps = log_prob_comp.sum(dim=-1)

        log_weights = F.log_softmax(self.mixture_logits, dim=-1)
        log_weights = log_weights.expand(log_prob_comps.shape)

        log_prob = torch.logsumexp(log_weights + log_prob_comps, dim=-1)

        return log_prob

    @property
    def mean(self):
        """
        Mixture mean: weighted sum of component means
        """
        weights = F.softmax(self.mixture_logits, dim=-1)

        return torch.sum(weights.unsqueeze(-1) * self.means, dim=-2)

    def variance(self):
        """
        Mixture variance: weighted sum of (variance + squared mean) minus squared mixture mean
        """
        weights = F.softmax(self.mixture_logits, dim=-1)
        mixture_mean = self.mean

        comp_var = self.stds**2
        second_moment = torch.sum(
            weights.unsqueeze(-1) * (comp_var + self.means**2), dim=-2
        )

        return second_moment - mixture_mean**2

    def entropy(self):

        raise NotImplementedError(
            "Entropy is not implemented in Mixture of Gaussians distribution."
        )


# In order to register the kd.kl_divergence() function for the MixtureOfGaussians class
@td.kl.register_kl(MixtureOfGaussians, MixtureOfGaussians)
def kl_mog_mog(p, q):
    # Monte Carlo sampling from p
    num_samples = 5000
    samples = p.rsample((num_samples,))
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)

    kl = (log_p - log_q).mean(dim=0)
    return kl


# ---------- VAE CLASSES DEFINITIONS ---------- #


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder, beta=1.0):
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
        self.beta = beta

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
        z = q.rsample()
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()),
            dim=0,
        )

        return elbo

    def kl_div_loss(self, x):
        q = self.encoder(x)
        z = q.rsample()
        kl_loss = torch.mean(
            self.beta * td.kl_divergence(q, self.prior()),
            dim=0,
        )
        return kl_loss

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

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        z = q.rsample()
        return z

    def pca_posterior(self, x):
        """
        Given a set of points, compute the PCA of the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        z = self.get_posterior(x)
        pca = PCA(n_components=2)
        return pca.fit_transform(z.detach().cpu())

    def pca_prior(self, n_samples):
        """
        Given a number of samples, get the PCA from the sampling of the prior distribution.
        """
        samples = self.prior().sample(torch.Size([n_samples]))
        pca = PCA(n_components=2)
        return pca.fit_transform(samples.detach().cpu())


class ConditionalVAE(nn.Module):
    """
    Define a conditional Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder, beta=1.0):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(ConditionalVAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta

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
        z = q.rsample()
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()),
            dim=0,
        )

        return elbo

    def kl_div_loss(self, x):
        q = self.encoder(x)
        z = q.rsample()
        kl_loss = torch.mean(
            self.beta * td.kl_divergence(q, self.prior()),
            dim=0,
        )
        return kl_loss

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
        # Obtain posterior and sample
        q_zx = self.encoder(x)
        z = q_zx.rsample()

        # Forward pass to the decoder
        p_xz = self.decoder(z)

        # Loss function
        log_q_zx = q_zx.log_prob(z)
        log_p_z = self.log_prob(z)

        recon_term = p_xz.log_prob(x).mean()
        kl_term = self.beta * (log_q_zx - log_p_z).mean()

        return -recon_term + kl_term

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        z = q.rsample()
        return z

    def log_prob(self, z):

        return self.prior().log_prob(z)

    def pca_posterior(self, x):
        """
        Given a set of points, compute the PCA of the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        z = self.get_posterior(x)
        pca = PCA(n_components=2)
        return pca.fit_transform(z.detach().cpu())

    def pca_prior(self, n_samples):
        """
        Given a number of samples, get the PCA from the sampling of the prior distribution.
        """
        samples = self.prior().sample(torch.Size([n_samples]))
        pca = PCA(n_components=2)
        return pca.fit_transform(samples.detach().cpu())


class ConditionalEnsembleVAE(nn.Module):
    """
    Define a conditional Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoders: nn.ModuleList, encoder, beta=1.0):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.ModuleList]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(ConditionalVAE, self).__init__()
        self.prior = prior
        self.decoders = decoders
        self.encoder = encoder
        self.beta = beta

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
        z = q.rsample()
        elbo = 0

        for decoder in self.decoders:
            elbo += torch.mean(
                decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()),
                dim=0,
            )

        return elbo / len(self.decoders)

    def kl_div_loss(self, x):
        q = self.encoder(x)
        z = q.rsample()
        kl_loss = torch.mean(
            self.beta * td.kl_divergence(q, self.prior()),
            dim=0,
        )
        return kl_loss

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        sample = []
        for decoder in self.decoders:
            sample.append(decoder(z).sample())
        return torch.stack(sample, dim=0).float().mean(dim=0).floor().int()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        # Obtain posterior and sample
        q_zx = self.encoder(x)
        z = q_zx.rsample()

        # Forward pass to the decoder)

        # Loss function
        log_q_zx = q_zx.log_prob(z)
        log_p_z = self.log_prob(z)

        recon_term = 0

        for decoder in self.decoders:
            p_xz = self.decoder(z)

            recon_term += p_xz.log_prob(x).mean()

        kl_term = self.beta * (log_q_zx - log_p_z).mean()

        return -(recon_term / len(self.decoders)) + kl_term

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        z = q.rsample()
        return z

    def log_prob(self, z):

        return self.prior().log_prob(z)

    def pca_posterior(self, x):
        """
        Given a set of points, compute the PCA of the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        z = self.get_posterior(x)
        pca = PCA(n_components=2)
        return pca.fit_transform(z.detach().cpu())

    def pca_prior(self, n_samples):
        """
        Given a number of samples, get the PCA from the sampling of the prior distribution.
        """
        samples = self.prior().sample(torch.Size([n_samples]))
        pca = PCA(n_components=2)
        return pca.fit_transform(samples.detach().cpu())


class VampPriorMixtureConditionalVAE(nn.Module):
    """
    Define a VampPrior Variational Autoencoder model.
    """

    def __init__(
        self,
        encoder,
        decoder,
        input_dim,
        batch_dim,
        n_comps,
        d_z,
        beta=1.0,
        data_loader=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim
        self.batch_dim = batch_dim
        self.K = n_comps
        self.d_z = d_z
        self.beta = beta

        # Pseudo-inputs
        if data_loader is not None:
            self.u = self.sample_from_dataloader(data_loader)
        else:
            self.u = nn.Parameter(
                torch.cat(
                    [torch.rand(n_comps, input_dim), torch.zeros(n_comps, batch_dim)],
                    dim=1,
                )
            )

        # MoG prior parameters besides location
        self.prior_var = nn.Parameter(torch.randn(n_comps, self.d_z))
        self.prior_pi = nn.Parameter(torch.zeros(n_comps))

    def sample_from_dataloader(self, data_loader):
        # all_data = []
        # bio_label = []
        # for batch in data_loader:
        #     x = batch[0]
        #     z = batch[2]  # biological variability
        #     all_data.append(x)
        #     bio_label.append(z)
        #     if len(all_data) * x.shape[0] >= self.K:
        #         break

        # all_data = torch.cat(all_data, dim=0)
        # bio_label = torch.cat(bio_label, dim=0)
        # bio_dict = torch.unique(bio_label, dim=0)

        # selected_u = []

        # for label in bio_dict:
        #     for i in range(all_data.shape[0]):
        #         if torch.equal(bio_label[i], label):
        #             selected_u.append(all_data[i])
        #             break

        # selected_u = torch.stack(selected_u)
        # selected_u = torch.cat([selected_u, torch.zeros(self.K, self.batch_dim)], dim=1)
        # return nn.Parameter(selected_u.clone().detach().requires_grad_(True))

        all_data = []
        bio_label = []
        # Collect until we have at least K samples
        for batch in data_loader:
            x = batch[0]
            z = batch[2]  # biological variability
            all_data.append(x)
            bio_label.append(z)
            if len(all_data) * x.shape[0] >= self.K:
                break

        all_data = torch.cat(all_data, dim=0)  # (N, D)
        bio_label = torch.cat(bio_label, dim=0)  # (N, L)

        # Find the unique labels
        bio_dict = torch.unique(bio_label, dim=0)  # (G, L), G = #groups

        # For each unique label, compute the mean of its members
        selected_u = []
        for label in bio_dict:
            # mask of samples matching this label
            mask = (bio_label == label).all(dim=1)  # (N,)
            group_data = all_data[mask]  # (n_i, D)
            group_mean = group_data.mean(dim=0)  # (D,)
            selected_u.append(group_mean)

        selected_u = torch.stack(selected_u, dim=0)  # (G, D)

        # Zero pad for batch dim
        zeros_pad = torch.zeros(self.K, self.batch_dim, device=selected_u.device)
        selected_u = torch.cat([selected_u, zeros_pad], dim=1)

        # Return as a learnable parameter
        return nn.Parameter(selected_u.clone().detach().requires_grad_(True))

    def get_prior(self):
        # encode pseudo inputs
        mog_u = self.encoder(self.u)

        # sample from encoded distribution to compute prior components centroids
        mu_u = mog_u.rsample()

        # compute prior
        w = self.prior_pi.view(-1)
        stds = torch.sqrt(torch.nn.functional.softplus(self.prior_var) + 1e-8)
        prior = MixtureOfGaussians(w, mu_u, stds)

        return prior

    def sample(self, n_samples):

        prior = self.get_prior()

        return prior.rsample(sample_shape=torch.Size([n_samples]))

    def pca_prior(self, n_samples):
        """
        Given a number of samples, get the PCA from the sampling of the prior distribution.
        """
        samples = self.sample(n_samples)
        pca = PCA(n_components=2)
        return pca.fit_transform(samples.detach().cpu())

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        z = q.rsample()
        return z

    def pca_posterior(self, x):
        """
        Given a set of points, compute the PCA of the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        z = self.get_posterior(x)
        pca = PCA(n_components=2)
        return pca.fit_transform(z.detach().cpu())

    def log_prob(self, z):

        prior = self.get_prior()

        return prior.log_prob(z)

    def forward(self, x):
        # Obtain posterior and sample
        q_zx = self.encoder(x)
        z = q_zx.rsample()

        # Forward pass to the decoder
        p_xz = self.decoder(z)

        # Loss function
        log_q_zx = q_zx.log_prob(z)
        log_p_z = self.log_prob(z)

        recon_term = p_xz.log_prob(x).mean()
        kl_term = self.beta * (log_q_zx - log_p_z).mean()

        return -recon_term + kl_term


class VampPriorMixtureConditionalEnsembleVAE(nn.Module):
    """
    Define a VampPrior Variational Autoencoder model.
    """

    def __init__(
        self,
        encoder,
        decoders: nn.ModuleList,
        input_dim,
        batch_dim,
        n_comps,
        d_z,
        beta=1.0,
        data_loader=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.input_dim = input_dim
        self.batch_dim = batch_dim
        self.K = n_comps
        self.d_z = d_z
        self.beta = beta

        # Pseudo-inputs
        if data_loader is not None:
            self.u = self.sample_from_dataloader(data_loader)
        else:
            self.u = nn.Parameter(
                torch.cat(
                    [torch.rand(n_comps, input_dim), torch.zeros(n_comps, batch_dim)],
                    dim=1,
                )
            )

        # MoG prior parameters besides location
        self.prior_var = nn.Parameter(torch.randn(n_comps, self.d_z))
        self.prior_pi = nn.Parameter(torch.zeros(n_comps))

    def sample_from_dataloader(self, data_loader):
        all_data = []
        bio_label = []
        for batch in data_loader:
            x = batch[0]
            z = batch[2]  # biological variability
            all_data.append(x)
            bio_label.append(z)
            if len(all_data) * x.shape[0] >= self.K:
                break

        all_data = torch.cat(all_data, dim=0)
        bio_label = torch.cat(bio_label, dim=0)
        bio_dict = torch.unique(bio_label, dim=0)

        selected_u = []

        for label in bio_dict:
            for i in range(all_data.shape[0]):
                if torch.equal(bio_label[i], label):
                    selected_u.append(all_data[i])
                    break

        selected_u = torch.stack(selected_u)
        selected_u = torch.cat([selected_u, torch.zeros(self.K, self.batch_dim)], dim=1)
        return nn.Parameter(selected_u.clone().detach().requires_grad_(True))

    def get_prior(self):
        # encode pseudo inputs
        mog_u = self.encoder(self.u)

        # sample from encoded distribution to compute prior components centroids
        mu_u = mog_u.rsample()

        # compute prior
        w = self.prior_pi.view(-1)
        stds = torch.sqrt(torch.nn.functional.softplus(self.prior_var) + 1e-8)
        prior = MixtureOfGaussians(w, mu_u, stds)

        return prior

    def sample(self, n_samples):

        prior = self.get_prior()

        return prior.rsample(sample_shape=torch.Size([n_samples]))

    def pca_prior(self, n_samples):
        """
        Given a number of samples, get the PCA from the sampling of the prior distribution.
        """
        samples = self.sample(n_samples)
        pca = PCA(n_components=2)
        return pca.fit_transform(samples.detach().cpu())

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        z = q.rsample()
        return z

    def pca_posterior(self, x):
        """
        Given a set of points, compute the PCA of the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        z = self.get_posterior(x)
        pca = PCA(n_components=2)
        return pca.fit_transform(z.detach().cpu())

    def log_prob(self, z):

        prior = self.get_prior()

        return prior.log_prob(z)

    def forward(self, x):
        # Obtain posterior and sample
        q_zx = self.encoder(x)
        z = q_zx.rsample()

        # Loss function
        log_q_zx = q_zx.log_prob(z)
        log_p_z = self.log_prob(z)

        # Forward pass to the decoder
        recon_term = 0
        for decoder in self.decoders:
            p_xz = self.decoder(z)
            recon_term += p_xz.log_prob(x).mean()

        kl_term = self.beta * (log_q_zx - log_p_z).mean()

        return -(recon_term / len(self.decoders)) + kl_term


# ---------- DISCRIMINATOR AND CONTRASTIVE LEARNING ---------- #


class BatchDiscriminator(nn.Module):
    """
    Define the Batch Discriminator for adversarial training
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        batch_class = self.net(x)
        return batch_class

    def loss(self, pred, true):

        loss = nn.CrossEntropyLoss()

        return loss(pred, true)


class SupervisedContrastiveLoss(nn.Module):
    """
    Contrastive loss definition
    """

    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp

    def forward(self, latent_points, labels):
        """
        latent_points: [batch_size, d_z]
        labels: [batch_size]
        """
        B, d_z = latent_points.shape
        # Normalizing to unit length
        embeddings = F.normalize(latent_points, dim=1)
        # Cosine similarity matrix
        sim_matrix = (
            torch.matmul(embeddings, embeddings.T) / self.temp
            - torch.eye(B, device=latent_points.device) * 1e-9
        )
        # Masking [i, j] = 1 if i and j share label
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)
        # Compute log-sum-exp over all except self
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Compute loss
        loss = -mean_log_prob.mean()
        return loss


# ---------- TRAINING LOOP ---------- #


def adversarial_loss(
    pred_logits: torch.Tensor,
    true_labels: torch.Tensor,
    loss_type: str = "CrossEntropy",
):
    """
    Compute adversarial loss for generator (to fool discriminator).

    Args:
        pred_logits: [batch_size, n_classes] raw discriminator outputs.
        true_labels: [batch_size] long tensor of class indices.
        loss_type: "CrossEntropy" or "Uniform".

    Returns:
        A scalar loss (negated for generator on CrossEntropy).
    """
    if loss_type == "CrossEntropy":
        ce = F.cross_entropy(pred_logits, true_labels)
        return -ce

    elif loss_type == "Uniform":
        log_probs = F.log_softmax(pred_logits, dim=1)
        target = torch.full_like(log_probs, 1.0 / pred_logits.size(1))
        return F.kl_div(log_probs, target, reduction="batchmean")

    else:
        raise ValueError(f"Unsupported adversarial loss type: {loss_type}")


def pre_train_abaco(
    vae,
    vae_optim_pre,
    discriminator,
    disc_optim,
    adv_optim,
    data_loader,
    epochs,
    device,
    w_contra=1.0,
    temp=0.1,
    w_elbo=1.0,
    w_disc=1.0,
    w_adv=1.0,
    disc_loss_type="CrossEntropy",
    n_disc_updates=1,
    label_smooth=0.1,
    normal=False,
    count=True,
):
    """
    Pre-training of conditional VAE with contrastive loss and adversarial mixing in latent space.
    """
    vae.train()
    discriminator.train()
    contra_criterion = SupervisedContrastiveLoss(temp)

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(
        range(total_steps),
        desc="Pre-training: VAE for reconstructing data and batch mixing adversarial training",
    )

    for epoch in range(epochs):

        for x, y_onehot, z_onehot in data_loader:
            # Move all tensors to the correct device
            x = x.to(device)
            if count == False:
                x_sum = x.sum(dim=1, keepdim=True)
                x = x / x_sum

            y_onehot = y_onehot.to(device)  # Batch one hot label
            z_onehot = z_onehot.to(device)  # Bio one hot label
            y_idx = y_onehot.argmax(1)
            z_idx = z_onehot.argmax(1)

            # === Step 1: Discriminator on latent (freeze encoder) ===
            with torch.no_grad():
                if normal == False:
                    pi, mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                    mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)
                    d_input = torch.cat([mu_bar, z_onehot], dim=1)

                else:
                    mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                    d_input = torch.cat([mu, z_onehot], dim=1)

            for _ in range(n_disc_updates):
                disc_optim.zero_grad()
                logits = discriminator(d_input)
                loss_disc = w_disc * F.cross_entropy(
                    logits, y_idx, label_smoothing=label_smooth
                )
                loss_disc.backward()
                disc_optim.step()

            # === Step 2: Adversarial update on encoder ===
            adv_optim.zero_grad()
            if normal == False:
                pi, mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)
                logits_fake = discriminator(torch.cat([mu_bar, z_onehot], dim=1))
            else:
                mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                logits_fake = discriminator(torch.cat([mu, z_onehot], dim=1))
            loss_adv = w_adv * adversarial_loss(
                pred_logits=logits_fake, true_labels=y_idx, loss_type=disc_loss_type
            )
            loss_adv.backward()
            adv_optim.step()

            # === Step 3: VAE reconstruction + contrastive ===
            vae_optim_pre.zero_grad()
            q_zx = vae.encoder(torch.cat([x, y_onehot], dim=1))
            latent = q_zx.rsample()
            p_xz = vae.decoder(torch.cat([latent, y_onehot], dim=1))

            recon_term = p_xz.log_prob(x).mean()
            kl_beta = vae.beta * max(0.0, (epoch / epochs))
            kl_term = kl_beta * (q_zx.log_prob(latent) - vae.log_prob(latent)).mean()
            elbo_loss = -(recon_term - kl_term)
            contra_loss = w_contra * contra_criterion(latent, z_idx)

            total_loss = w_elbo * elbo_loss + contra_loss
            total_loss.backward()
            vae_optim_pre.step()

            # Update progress bar
            progress_bar.set_postfix(
                elbo=f"{elbo_loss.item():.4f}",
                contra=f"{contra_loss.item():.4f}",
                disc=f"{loss_disc.item():.4f}",
                adv=f"{loss_adv.item():.4f}",
                epoch=f"{epoch}/{epochs+1}",
            )
            progress_bar.update()

    progress_bar.close()


def train_abaco(
    vae,
    vae_optim_post,
    data_loader,
    epochs,
    device,
    w_elbo=1.0,
    w_cycle=1.0,
    cycle="KL",
    smooth_annealing=False,
):
    """
    This function trains a pre-trained ABaCo cVAE decoder but applies masking to batch labels so
    information passed solely depends on the latent space which had batch mixing
    """

    vae.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(
        range(total_steps), desc="Training: VAE decoder with masked batch labels"
    )

    for epoch in range(epochs):
        # Introduce slow transition to full batch masking
        if smooth_annealing:
            alpha = max(0.0, 1.0 - (2 * epoch / epochs))
        else:
            alpha = 0.0

        data_iter = iter(data_loader)
        for loader_data in data_iter:

            x = loader_data[0].to(device)
            y = loader_data[1].to(device).float()  # Batch label
            z = loader_data[2].to(device).float()  # Bio type label

            # VAE ELBO computation with masked batch label
            vae_optim_post.zero_grad()

            # Forward pass to encoder
            q_zx = vae.encoder(torch.cat([x, y], dim=1))

            # Sample from encoded point
            latent_points = q_zx.rsample()

            # Forward pass to the decoder
            p_xz = vae.decoder(torch.cat([latent_points, alpha * y], dim=1))

            # Log probabilities of prior and posterior
            log_q_zx = q_zx.log_prob(latent_points)
            log_p_z = vae.log_prob(latent_points)

            # Compute ELBO
            recon_term = p_xz.log_prob(x).mean()
            # kl_term = vae.beta * (log_q_zx - log_p_z).mean()
            elbo = recon_term

            # Compute loss
            elbo_loss = -elbo

            # Compute overall loss and backprop
            recon_loss = w_elbo * elbo_loss

            # Latent cycle step: regularization term for demostrating encoded reconstructed point = encoded original point

            if cycle == "MSE":
                # Original encoded point
                pi, mu, _ = vae.encoder.encode(torch.cat([x, y], dim=1))
                mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)

                # Reconstructed encoded point
                x_r = p_xz.sample()
                pi_r, mu_r, _ = vae.encoder.encode(torch.cat([x_r, y], dim=1))
                mu_r_bar = (mu_r * pi_r.unsqueeze(2)).sum(dim=1)

                # Backpropagation
                cycle_loss = F.mse_loss(mu_r_bar, mu_bar)

            elif cycle == "CE":
                # Original encoded point - mixing probability
                pi, _, _ = vae.encoder.encode(torch.cat([x, y], dim=1))

                # Reconstructed encoded point - mixing probability
                x_r = p_xz.sample()
                pi_r, _, _ = vae.encoder.encode(torch.cat([x_r, y], dim=1))

                # Backpropagation
                cycle_loss = F.cross_entropy(pi_r, pi)

            elif cycle == "KL":
                # Original encoded point - pdf
                q_zx = vae.encoder(torch.cat([x, y], dim=1))

                # Reconstructed encoded point - pdf
                x_r = p_xz.sample()
                q_zx_r = vae.encoder(torch.cat([x_r, y], dim=1))

                # Backpropagation
                cycle_loss = torch.mean(
                    td.kl_divergence(q_zx_r, q_zx),
                    dim=0,
                )

            else:
                cycle_loss = 0

            vae_loss = recon_loss + w_cycle * cycle_loss
            vae_loss.backward()
            vae_optim_post.step()

            # Update progress bar
            progress_bar.set_postfix(
                vae_loss=f"{vae_loss.item():12.4f}",
                epoch=f"{epoch+1}/{epochs}",
            )
            progress_bar.update()

    progress_bar.close()


# Other functions


def contour_plot(samples, n_levels=10, x_offset=5, y_offset=5, alpha=0.8):
    """
    Given an array computes the contour plot

    Parameters:
        samples: [np.array]
            An array with (,2) dimensions
    """
    x = samples[:, 0]
    y = samples[:, 1]

    x_min, x_max = x.min() - x_offset, x.max() + x_offset
    y_min, y_max = y.min() - y_offset, y.max() + y_offset
    x_grid = np.linspace(x_min, x_max, 500)
    y_grid = np.linspace(y_min, y_max, 500)
    X, Y = np.meshgrid(x_grid, y_grid)

    kde = gaussian_kde(samples.T)
    Z = kde(np.vstack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)

    contour = plt.contourf(X, Y, Z, levels=n_levels, alpha=alpha)

    return contour


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    # Compute constant term outside the exponentiation
    const = torch.log(torch.tensor(2 * math.pi, device=x.device))

    # Compute log probability
    log_p = -0.5 * (const + log_var + torch.exp(-log_var) * (x - mu) ** 2)

    if dim is None:
        dim = -1
    log_p = log_p.sum(dim=dim)

    if reduction == "avg":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


# Baseline


def abaco_run(
    dataloader: torch.utils.data.DataLoader,
    n_batches: int,
    n_bios: int,
    device: torch.device,
    input_size: int,
    new_pre_train: bool = False,
    seed: int = 42,
    d_z: int = 16,
    prior: str = "VMM",
    count: bool = True,
    pre_epochs:int = 2000,
    post_epochs: int = 2000,
    kl_cycle: bool = True,
    smooth_annealing:bool = False,
    # VAE Model architecture
    encoder_net: list = [1024, 512, 256],
    decoder_net: list = [256, 512, 1024],
    vae_act_func = nn.ReLU(),
    # Discriminator architecture
    disc_net: list = [256, 128, 64],
    disc_act_func=nn.ReLU(),
    disc_loss_type: str = "CrossEntropy",
    # Model weights
    w_elbo: float = 1.0,
    beta: float = 20.0,
    w_disc: float = 1.0,
    w_adv: float = 1.0,
    w_contra: float = 10.0,
    temp: float = 0.1,
    w_cycle: float = 0.1,
    # Learning rates
    vae_pre_lr: float = 1e-3,
    vae_post_lr: float = 1e-4,
    disc_lr: float = 1e-5,
    adv_lr: float =1e-5,
):
    """
    Function to run the ABaCo model training.

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        Pytorch dataLoader for the training data.
    n_batches: int
        Number of batches in the dataset. For example, if samples were sequenced in
        5 batches (e.g., 5 different dates) then batches = 5.
    n_bios: int
        Number of labels or (potential) clusters based on biological variance. For example, if 2 experimental
        conditions (e.g., control and treatment) then n_bios = 2.
    device: torch.device
        Device to run the model on, e.g., "cuda" or "cpu".
    input_size: int
        Number of features in the input data, columns. For example, if the input is a gene expression matrix with 1000 genes,
        then input_size = 1000.
    new_pre_train: bool
        If True, use the new pre-training method with adversarial training and contrastive loss.
    seed: int
        Random seed for reproducibility.
    d_z: int
        Dimensionality of the latent space. For example, if d_z = 16, then the latent space will have 16 dimensions.
    prior: str
        Prior distribution used. Baseline is "VMM" (VampPrior Mixture Model). Options are "VMM" 
        "MoG" (Mixture of Gaussians), or "Normal".
    count: bool
        If True, the model will use a zero-inflated negative binomial (ZINB) decoder.
        If False, it will use a zero-inflated Dirichlet (ZIDirichlet) decoder.
    pre_epochs: int
        Number of epochs for first phase of ABaCo: data reconstruction. Default is 2000.
    post_epochs: int
        Number of epochs for second phase of ABaCo: batch correction. Default is 2000.
    kl_cycle: bool
        If True, the model will use a KL divergence cycle loss during second phase of ABaCo (batch correction).
        If False, cross loss 0 
    smooth_annealing: bool
        Slow batch masking during ABaCo second phase (batch correction) to avoid exploding gradients. 
        Default is False.
    encoder_net: list
        List of integers defining the architecture of the encoder. Each integer is a layer size.
        For example, [1024, 512, 256] means the encoder will have three layers with 1024, 512, and 256 neurons respectively.
    decoder_net: list
        List of integers defining the architecture of the decoder. Each integer is a layer size.
        For example, [256, 512, 1024] means the decoder will have three layers with 256, 512, and 1024 neurons respectively.
    vae_act_func: nn.Module
        Activation function for the VAE encoder and decoder. Default is nn.ReLU().
    disc_net: list
        List of integers defining the architecture of the discriminator. Each integer is a layer size.
        For example, [256, 128, 64] means the discriminator will have three layers with 256, 128, and 64 neurons respectively.
    disc_act_func: nn.Module
        Activation function for the discriminator. Default is nn.ReLU().
    disc_loss_type: str
        Type of loss function for the discriminator. Options are "CrossEntropy" or "Uniform".
        Default is "CrossEntropy".
    w_elbo: float
        Weight of the ELBO loss in the pre-training phase. Default is 1.0
    beta: float
        KL-divergence coefficient: higher value yields bigger penalization from the prior distribution
        during training. Default is 20.0.
    w_disc: float
        Weight of the discriminator loss in the pre-training phase. Default is 1.0
    w_adv: float
        Weight of the adversarial loss in the pre-training phase. Default is 1.0 
    w_contra: float
        Contrastive learning power. Higher value yields a higher separation of biological groups 
        at the latent space. Sometimes higher is better.
    w_cycle: float
        Higher value leads to more unstability during ABaCo second phase. Default is 0.1.
    temp: float
        Temperature for the contrastive loss. Default is 0.1.
    vae_pre_lr: float
        Learning rate for first phase of ABaCo. Default is 1e-3.
    vae_post_lr: float
        Learning rate for second phase (batch correction). Default is 1e-4.
    disc_lr: float
        Learning rate for the batch discriminator. Default is 1e-5.
    adv_lr: float
        Adversarial learning rate: to the encoder, if batch effect is on latent space. Default is 1e-5.

    Returns
    -------
    vae: 
        Trained ABaCo model


    """
    # Number of biological groups
    K = n_bios
    # Number of batches
    n_batches = n_batches
    # Default Normal
    normal = False
    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Defining the VAE architecture
    if prior == "VMM":
        # Defining Encoder
        encoder_net = [input_size + n_batches] + encoder_net  # first layer: conditional
        encoder_net.append(K * (2 * d_z + 1))  # last layer
        modules = []
        for i in range(len(encoder_net) - 1):
            modules.append(nn.Linear(encoder_net[i], encoder_net[i + 1]))
            modules.append(vae_act_func)
        modules.pop()  # Drop last activation function
        encoder = MoGEncoder(nn.Sequential(*modules), n_comp=K)

        # Defining Decoder
        if count:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(3 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function

            decoder = ZINBDecoder(nn.Sequential(*modules))
        else:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(2 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function
            decoder = ZIDirichletDecoder(nn.Sequential(*modules))

        # Defining VAE
        vae = VampPriorMixtureConditionalVAE(
            encoder=encoder,
            decoder=decoder,
            input_dim=input_size,
            n_comps=K,
            batch_dim=n_batches,
            d_z=d_z,
            beta=beta,
            data_loader=dataloader,
        ).to(device)

        # Defining VAE optims
        vae_optim_pre = torch.optim.Adam(
            [
                {"params": vae.encoder.parameters()},
                {"params": vae.decoder.parameters()},
                {"params": [vae.u, vae.prior_pi, vae.prior_var]},
            ],
            lr=vae_pre_lr,
        )
        vae_optim_post = torch.optim.Adam(
            [
                {"params": vae.decoder.parameters()},
            ],
            lr=vae_post_lr,
        )

    elif prior == "MoG":
        # Defining Encoder
        encoder_net = [input_size + n_batches] + encoder_net  # first layer: conditional
        encoder_net.append(K * (2 * d_z + 1))  # last layer
        modules = []
        for i in range(len(encoder_net) - 1):
            modules.append(nn.Linear(encoder_net[i], encoder_net[i + 1]))
            modules.append(vae_act_func)
        modules.pop()  # Drop last activation function
        encoder = MoGEncoder(nn.Sequential(*modules), n_comp=K)

        # Defining Decoder
        if count:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(3 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function

            decoder = ZINBDecoder(nn.Sequential(*modules))
        else:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(2 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function
            decoder = ZIDirichletDecoder(nn.Sequential(*modules))

        # Defining prior
        prior = MoGPrior(d_z, K)

        # Defining VAE
        vae = ConditionalVAE(
            prior=prior,
            encoder=encoder,
            decoder=decoder,
            beta=beta,
        ).to(device)

        # Defining VAE optims

        vae_optim_pre = torch.optim.Adam(vae.parameters(), lr=vae_pre_lr)

        vae_optim_post = torch.optim.Adam(
            [
                {"params": vae.decoder.parameters()},
            ],
            lr=vae_post_lr,
        )

    elif prior == "Normal":
        # change normal variable
        normal = True
        # Defining Encoder
        encoder_net = [input_size + n_batches] + encoder_net  # first layer: conditional
        encoder_net.append(2 * d_z)  # last layer
        modules = []
        for i in range(len(encoder_net) - 1):
            modules.append(nn.Linear(encoder_net[i], encoder_net[i + 1]))
            modules.append(vae_act_func)
        modules.pop()  # Drop last activation function
        encoder = NormalEncoder(nn.Sequential(*modules))

        # Defining Decoder
        if count:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(3 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function

            decoder = ZINBDecoder(nn.Sequential(*modules))
        else:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(2 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function
            decoder = ZIDirichletDecoder(nn.Sequential(*modules))

        # Defining prior
        prior = NormalPrior(d_z)

        # Defining VAE
        vae = ConditionalVAE(
            prior=prior,
            encoder=encoder,
            decoder=decoder,
            beta=beta,
        ).to(device)

        # Defining VAE optims

        vae_optim_pre = torch.optim.Adam(vae.parameters(), lr=vae_pre_lr)

        vae_optim_post = torch.optim.Adam(
            [
                {"params": vae.decoder.parameters()},
            ],
            lr=vae_post_lr,
        )

    else:
        raise ValueError(f"Prior distribution '{prior}' isn't a valid option: 'VMM', 'MoG' or 'Normal'")

    # Defining the batch discriminator architecture
    disc_net = [d_z + K] + disc_net  # first layer: conditional
    disc_net.append(n_batches)  # last layer
    modules = []
    for i in range(len(disc_net) - 1):
        modules.append(nn.Linear(disc_net[i], disc_net[i + 1]))
        modules.append(disc_act_func)
    modules.pop()  # remove last activation function
    discriminator = BatchDiscriminator(nn.Sequential(*modules)).to(device)

    # Defining the batch discriminator optimizers

    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=disc_lr)

    adv_optim = torch.optim.Adam(vae.encoder.parameters(), lr=adv_lr)

    # FIRST STEP: TRAIN VAE MODEL TO RECONSTRUCT DATA AND BATCH MIXING OF LATENT SPACE
    if new_pre_train == False:
        pre_train_abaco(
            vae=vae,
            vae_optim_pre=vae_optim_pre,
            discriminator=discriminator,
            disc_optim=disc_optim,
            adv_optim=adv_optim,
            data_loader=dataloader,
            epochs=pre_epochs,
            device=device,
            w_elbo=w_elbo,
            w_contra=w_contra,
            temp=temp,
            w_adv=w_adv,
            w_disc=w_disc,
            disc_loss_type=disc_loss_type,
            normal=normal,
            count=count,
        )

    else:
        new_pre_train_abaco(
            vae=vae,
            vae_optim_pre=vae_optim_pre,
            discriminator=discriminator,
            disc_optim=disc_optim,
            adv_optim=adv_optim,
            data_loader=dataloader,
            normal_epochs=pre_epochs,
            mog_epochs=pre_epochs,
            device=device,
            w_elbo=w_elbo,
            w_contra=w_contra,
            temp=temp,
            w_adv=w_adv,
            w_disc=w_disc,
            disc_loss_type=disc_loss_type,
            normal=normal,
            count=count,
        )

    # SECOND STEP: TRAIN DECODER TO PERFORM BATCH MIXING AT THE MODEL OUTPUT

    if kl_cycle:
        train_abaco(
            vae=vae,
            vae_optim_post=vae_optim_post,
            data_loader=dataloader,
            epochs=post_epochs,
            device=device,
            w_elbo=w_elbo,
            w_cycle=w_cycle,
            cycle="KL",
            smooth_annealing=smooth_annealing,
        )

    else:
        train_abaco(
            vae=vae,
            vae_optim_post=vae_optim_post,
            data_loader=dataloader,
            epochs=post_epochs,
            device=device,
            w_elbo=w_elbo,
            w_cycle=0.0,
            cycle="None",
            smooth_annealing=smooth_annealing,
        )

    return vae


def abaco_recon(
    model,
    device,
    data,
    dataloader,
    sample_label,
    batch_label,
    bio_label,
    seed=42,
    det_encode=False,
    monte_carlo=100,
):
    """
    Function used to reconstruct data using trained ABaCo model.
    """
    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ohe_batch = one_hot_encoding(data[batch_label])[0]

    # Reconstructing data with trained model - DETERMINISTIC RECONSTRUCTION
    recon_data = []

    for x in dataloader:
        x = x[0].to(device)
        if det_encode == True:
            z = model.encoder.det_encode(torch.cat([x, ohe_batch.to(device)], dim=1))
        else:
            z = model.encoder.monte_carlo_encode(
                x=torch.cat([x, ohe_batch.to(device)], dim=1), K=monte_carlo
            )
        recon = model.decoder.monte_carlo_decode(
            z=torch.cat([z, torch.zeros_like(ohe_batch.to(device))], dim=1),
            K=monte_carlo,
        )
        recon_data.append(recon)

    np_recon_data = np.vstack([t.detach().cpu().numpy() for t in recon_data])

    otu_corrected_pd = pd.concat(
        [
            data[sample_label],
            data[batch_label],
            data[bio_label],
            pd.DataFrame(
                np_recon_data,
                index=data.index,
                columns=data.select_dtypes("number").columns,
            ),
        ],
        axis=1,
    )
    return otu_corrected_pd


# ------- ENSEMBLE MODEL: ONE ENCODER (ONE LATENT SPACE) -> MULTIPLE DECODERS


def pre_train_abaco_ensemble(
    vae,
    vae_optim_pre,
    discriminator,
    disc_optim,
    adv_optim,
    data_loader,
    epochs,
    device,
    w_contra=1.0,
    temp=0.1,
    w_elbo=1.0,
    w_disc=1.0,
    w_adv=1.0,
    disc_loss_type="CrossEntropy",
    n_disc_updates=1,
    label_smooth=0.1,
    normal=False,
    count=True,
):
    """
    Pre-training of conditional VAE with contrastive loss and adversarial mixing in latent space.
    """
    vae.train()
    discriminator.train()
    contra_criterion = SupervisedContrastiveLoss(temp)

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(
        range(total_steps),
        desc="Pre-training: VAE for reconstructing data and batch mixing adversarial training",
    )

    for epoch in range(epochs):

        for x, y_onehot, z_onehot in data_loader:
            # Move all tensors to the correct device
            x = x.to(device)
            if count == False:
                x_sum = x.sum(dim=1, keepdim=True)
                x = x / x_sum

            y_onehot = y_onehot.to(device)  # Batch one hot label
            z_onehot = z_onehot.to(device)  # Bio one hot label
            y_idx = y_onehot.argmax(1)
            z_idx = z_onehot.argmax(1)

            # === Step 1: Discriminator on latent (freeze encoder) ===
            with torch.no_grad():
                if normal == False:
                    pi, mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                    mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)
                    d_input = torch.cat([mu_bar, z_onehot], dim=1)

                else:
                    mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                    d_input = torch.cat([mu, z_onehot], dim=1)

            for _ in range(n_disc_updates):
                disc_optim.zero_grad()
                logits = discriminator(d_input)
                loss_disc = w_disc * F.cross_entropy(
                    logits, y_idx, label_smoothing=label_smooth
                )
                loss_disc.backward()
                disc_optim.step()

            # === Step 2: Adversarial update on encoder ===
            adv_optim.zero_grad()
            if normal == False:
                pi, mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)
                logits_fake = discriminator(torch.cat([mu_bar, z_onehot], dim=1))
            else:
                mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                logits_fake = discriminator(torch.cat([mu, z_onehot], dim=1))
            loss_adv = w_adv * adversarial_loss(
                pred_logits=logits_fake, true_labels=y_idx, loss_type=disc_loss_type
            )
            loss_adv.backward()
            adv_optim.step()

            # === Step 3: VAE reconstruction + contrastive ===
            vae_optim_pre.zero_grad()
            q_zx = vae.encoder(torch.cat([x, y_onehot], dim=1))
            latent = q_zx.rsample()
            recon_term = 0
            for decoder in vae.decoders:
                p_xz = decoder(torch.cat([latent, y_onehot], dim=1))
                recon_term += p_xz.log_prob(x).mean()

            kl_beta = vae.beta * max(
                0.0, 1.0 - (epoch / epochs)
            )  # smooth annealing for KL divergence ensures contrastive clustering early
            kl_term = kl_beta * (q_zx.log_prob(latent) - vae.log_prob(latent)).mean()
            elbo_loss = -(recon_term / len(vae.decoders) - kl_term)
            contra_loss = w_contra * contra_criterion(latent, z_idx)

            total_loss = w_elbo * elbo_loss + contra_loss
            total_loss.backward()
            vae_optim_pre.step()

            # Update progress bar
            progress_bar.set_postfix(
                elbo=f"{elbo_loss.item():.4f}",
                contra=f"{contra_loss.item():.4f}",
                disc=f"{loss_disc.item():.4f}",
                adv=f"{loss_adv.item():.4f}",
                epoch=f"{epoch}/{epochs+1}",
            )
            progress_bar.update()

    progress_bar.close()


def train_abaco_ensemble(
    vae,
    vae_optim_post,
    data_loader,
    epochs,
    device,
    w_elbo=1.0,
    w_cycle=1.0,
    cycle="KL",
    smooth_annealing=False,
):
    """
    This function trains a pre-trained ABaCo cVAE decoder but applies masking to batch labels so
    information passed solely depends on the latent space which had batch mixing
    """

    vae.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(
        range(total_steps), desc="Training: VAE decoder with masked batch labels"
    )

    for epoch in range(epochs):
        # Introduce slow transition to full batch masking
        if smooth_annealing:
            alpha = max(0.0, 1.0 - (2 * epoch / epochs))
        else:
            alpha = 0.0

        data_iter = iter(data_loader)
        for loader_data in data_iter:

            x = loader_data[0].to(device)
            y = loader_data[1].to(device).float()  # Batch label
            z = loader_data[2].to(device).float()  # Bio type label

            # VAE ELBO computation with masked batch label
            vae_optim_post.zero_grad()

            # Forward pass to encoder
            q_zx = vae.encoder(torch.cat([x, y], dim=1))

            # Sample from encoded point
            latent_points = q_zx.rsample()

            # Forward pass to the decoder
            recon_term = 0
            p_xzs = []
            for decoder in vae.decoders:
                p_xz = decoder(torch.cat([latent_points, alpha * y], dim=1))
                recon_term += p_xz.log_prob(x).mean()
                p_xzs.append(p_xz)

            # Log probabilities of prior and posterior
            log_q_zx = q_zx.log_prob(latent_points)
            log_p_z = vae.log_prob(latent_points)

            # Compute ELBO

            # kl_term = vae.beta * (log_q_zx - log_p_z).mean()
            elbo = recon_term / len(vae.decoders)

            # Compute loss
            elbo_loss = -elbo

            # Compute overall loss and backprop
            recon_loss = w_elbo * elbo_loss

            # Latent cycle step: regularization term for demostrating encoded reconstructed point = encoded original point

            if cycle == "MSE":
                # Original encoded point
                pi, mu, _ = vae.encoder.encode(torch.cat([x, y], dim=1))
                mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)

                # Reconstructed encoded point
                x_r = p_xz.sample()
                pi_r, mu_r, _ = vae.encoder.encode(torch.cat([x_r, y], dim=1))
                mu_r_bar = (mu_r * pi_r.unsqueeze(2)).sum(dim=1)

                # Backpropagation
                cycle_loss = F.mse_loss(mu_r_bar, mu_bar)

            elif cycle == "CE":
                # Original encoded point - mixing probability
                pi, _, _ = vae.encoder.encode(torch.cat([x, y], dim=1))

                # Reconstructed encoded point - mixing probability
                x_r = p_xz.sample()
                pi_r, _, _ = vae.encoder.encode(torch.cat([x_r, y], dim=1))

                # Backpropagation
                cycle_loss = F.cross_entropy(pi_r, pi)

            elif cycle == "KL":
                # Original encoded point - pdf
                q_zx = vae.encoder(torch.cat([x, y], dim=1))

                # Reconstructed encoded point - pdf
                x_rs = []
                for p_xz in p_xzs:
                    x_r = p_xz.sample()
                    x_rs.append(x_r)
                x_r = torch.stack(x_rs, dim=0).float().mean(dim=0).floor().int()
                q_zx_r = vae.encoder(torch.cat([x_r, y], dim=1))

                # Backpropagation
                cycle_loss = torch.mean(
                    td.kl_divergence(q_zx_r, q_zx),
                    dim=0,
                )

            else:
                cycle_loss = 0

            vae_loss = recon_loss + w_cycle * cycle_loss
            vae_loss.backward()
            vae_optim_post.step()

            # Update progress bar
            progress_bar.set_postfix(
                vae_loss=f"{vae_loss.item():12.4f}",
                epoch=f"{epoch+1}/{epochs}",
            )
            progress_bar.update()

    progress_bar.close()


def abaco_run_ensemble(
    dataloader,
    n_batches,
    n_bios,
    device,
    input_size,
    seed=42,
    d_z=16,
    prior="VMM",
    count=True,
    pre_epochs=2000,
    post_epochs=2000,
    kl_cycle=True,
    smooth_annealing=False,
    # VAE Model architecture
    encoder_net=[1024, 512, 256],
    n_dec=5,
    decoder_net=[256, 512, 1024],
    vae_act_func=nn.ReLU(),
    # Discriminator architecture
    disc_net=[256, 128, 64],
    disc_act_func=nn.ReLU(),
    disc_loss_type="CrossEntropy",
    # Model weights
    w_elbo=1.0,
    beta=20.0,
    w_disc=1.0,
    w_adv=1.0,
    w_contra=10.0,
    temp=0.1,
    # Learning rates
    vae_pre_lr=1e-3,
    vae_post_lr=1e-4,
    disc_lr=1e-5,
    adv_lr=1e-5,
):
    """Full ABaCo run with default setting"""
    # Number of biological groups
    K = n_bios
    # Number of batches
    n_batches = n_batches
    # Default Normal
    normal = False
    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Defining the VAE architecture
    if prior == "VMM":
        # Defining Encoder
        encoder_net = [input_size + n_batches] + encoder_net  # first layer: conditional
        encoder_net.append(K * (2 * d_z + 1))  # last layer
        modules = []
        for i in range(len(encoder_net) - 1):
            modules.append(nn.Linear(encoder_net[i], encoder_net[i + 1]))
            modules.append(vae_act_func)
        modules.pop()  # Drop last activation function
        encoder = MoGEncoder(nn.Sequential(*modules), n_comp=K)

        # Defining Decoder
        decoders = nn.ModuleList()
        if count:
            for _ in range(n_dec):
                decoder_net = [
                    d_z + n_batches
                ] + decoder_net  # first value: conditional
                decoder_net.append(3 * input_size)  # last layer
                modules = []
                for i in range(len(decoder_net) - 1):
                    modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                    modules.append(vae_act_func)
                modules.pop()  # Drop last activation function

                decoder = ZINBDecoder(nn.Sequential(*modules))
                decoders.append(decoder)
        else:
            for _ in range(n_dec):
                decoder_net = [
                    d_z + n_batches
                ] + decoder_net  # first value: conditional
                decoder_net.append(2 * input_size)  # last layer
                modules = []
                for i in range(len(decoder_net) - 1):
                    modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                    modules.append(vae_act_func)
                modules.pop()  # Drop last activation function
                decoder = ZIDirichletDecoder(nn.Sequential(*modules))
                decoders.append(decoder)

        # Defining VAE
        vae = VampPriorMixtureConditionalEnsembleVAE(
            encoder=encoder,
            decoders=decoders,
            input_dim=input_size,
            n_comps=K,
            batch_dim=n_batches,
            d_z=d_z,
            beta=beta,
            data_loader=dataloader,
        ).to(device)

        # Defining VAE optims
        vae_optim_pre = torch.optim.Adam(
            [
                {"params": vae.encoder.parameters()},
                {"params": vae.decoders.parameters()},
                {"params": [vae.u, vae.prior_pi, vae.prior_var]},
            ],
            lr=vae_pre_lr,
        )
        vae_optim_post = torch.optim.Adam(
            [
                {"params": vae.decoders.parameters()},
            ],
            lr=vae_post_lr,
        )

    elif prior == "MoG":
        # Defining Encoder
        encoder_net = [input_size + n_batches] + encoder_net  # first layer: conditional
        encoder_net.append(K * (2 * d_z + 1))  # last layer
        modules = []
        for i in range(len(encoder_net) - 1):
            modules.append(nn.Linear(encoder_net[i], encoder_net[i + 1]))
            modules.append(vae_act_func)
        modules.pop()  # Drop last activation function
        encoder = MoGEncoder(nn.Sequential(*modules), n_comp=K)

        # Defining Decoder
        if count:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(3 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function

            decoder = ZINBDecoder(nn.Sequential(*modules))
        else:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(2 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function
            decoder = ZIDirichletDecoder(nn.Sequential(*modules))

        # Defining prior
        prior = MoGPrior(d_z, K)

        # Defining VAE
        vae = ConditionalVAE(
            prior=prior,
            encoder=encoder,
            decoder=decoder,
            beta=beta,
        ).to(device)

        # Defining VAE optims

        vae_optim_pre = torch.optim.Adam(vae.parameters(), lr=vae_pre_lr)

        vae_optim_post = torch.optim.Adam(
            [
                {"params": vae.decoder.parameters()},
            ],
            lr=vae_post_lr,
        )

    elif prior == "Normal":
        # change normal variable
        normal = True
        # Defining Encoder
        encoder_net = [input_size + n_batches] + encoder_net  # first layer: conditional
        encoder_net.append(2 * d_z)  # last layer
        modules = []
        for i in range(len(encoder_net) - 1):
            modules.append(nn.Linear(encoder_net[i], encoder_net[i + 1]))
            modules.append(vae_act_func)
        modules.pop()  # Drop last activation function
        encoder = NormalEncoder(nn.Sequential(*modules))

        # Defining Decoder
        if count:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(3 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function

            decoder = ZINBDecoder(nn.Sequential(*modules))
        else:
            decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            decoder_net.append(2 * input_size)  # last layer
            modules = []
            for i in range(len(decoder_net) - 1):
                modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
                modules.append(vae_act_func)
            modules.pop()  # Drop last activation function
            decoder = ZIDirichletDecoder(nn.Sequential(*modules))

        # Defining prior
        prior = NormalPrior(d_z)

        # Defining VAE
        vae = ConditionalVAE(
            prior=prior,
            encoder=encoder,
            decoder=decoder,
            beta=beta,
        ).to(device)

        # Defining VAE optims

        vae_optim_pre = torch.optim.Adam(vae.parameters(), lr=vae_pre_lr)

        vae_optim_post = torch.optim.Adam(
            [
                {"params": vae.decoder.parameters()},
            ],
            lr=vae_post_lr,
        )

    else:
        raise ValueError(f"Prior distribution select isn't a valid option.")

    # Defining the batch discriminator architecture
    disc_net = [d_z + K] + disc_net  # first layer: conditional
    disc_net.append(n_batches)  # last layer
    modules = []
    for i in range(len(disc_net) - 1):
        modules.append(nn.Linear(disc_net[i], disc_net[i + 1]))
        modules.append(disc_act_func)
    modules.pop()  # remove last activation function
    discriminator = BatchDiscriminator(nn.Sequential(*modules)).to(device)

    # Defining the batch discriminator optimizers

    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=disc_lr)

    adv_optim = torch.optim.Adam(vae.encoder.parameters(), lr=adv_lr)

    # FIRST STEP: TRAIN VAE MODEL TO RECONSTRUCT DATA AND BATCH MIXING OF LATENT SPACE

    pre_train_abaco_ensemble(
        vae=vae,
        vae_optim_pre=vae_optim_pre,
        discriminator=discriminator,
        disc_optim=disc_optim,
        adv_optim=adv_optim,
        data_loader=dataloader,
        epochs=pre_epochs,
        device=device,
        w_elbo=w_elbo,
        w_contra=w_contra,
        temp=temp,
        w_adv=w_adv,
        w_disc=w_disc,
        disc_loss_type=disc_loss_type,
        normal=normal,
        count=count,
    )

    # SECOND STEP: TRAIN DECODER TO PERFORM BATCH MIXING AT THE MODEL OUTPUT

    if kl_cycle:
        train_abaco_ensemble(
            vae=vae,
            vae_optim_post=vae_optim_post,
            data_loader=dataloader,
            epochs=post_epochs,
            device=device,
            w_elbo=w_elbo,
            w_cycle=0.1,
            cycle="KL",
            smooth_annealing=smooth_annealing,
        )

    else:
        train_abaco_ensemble(
            vae=vae,
            vae_optim_post=vae_optim_post,
            data_loader=dataloader,
            epochs=post_epochs,
            device=device,
            w_elbo=w_elbo,
            w_cycle=0.0,
            cycle="None",
            smooth_annealing=smooth_annealing,
        )

    return vae


def abaco_recon_ensemble(
    model,
    device,
    data,
    dataloader,
    sample_label,
    batch_label,
    bio_label,
    seed=42,
    det_encode=False,
    monte_carlo=100,
):
    """
    Function used to reconstruct data using trained ABaCo model.
    """
    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ohe_batch = one_hot_encoding(data[batch_label])[0]

    # Reconstructing data with trained model - DETERMINISTIC RECONSTRUCTION
    recon_data = []

    for x in dataloader:
        x = x[0].to(device)
        if det_encode == True:
            z = model.encoder.det_encode(torch.cat([x, ohe_batch.to(device)], dim=1))
        else:
            z = model.encoder.monte_carlo_encode(
                x=torch.cat([x, ohe_batch.to(device)], dim=1), K=monte_carlo
            )
        recons = []
        for decoder in model.decoders:
            recon = decoder.monte_carlo_decode(
                z=torch.cat([z, torch.zeros_like(ohe_batch.to(device))], dim=1),
                K=monte_carlo,
            )
            recons.append(recon)
        recon = (
            torch.stack(recons, dim=0).float().mean(dim=0).floor().int()
        )  # added float() for being able to compute mean()
        recon_data.append(recon)

    np_recon_data = np.vstack([t.detach().cpu().numpy() for t in recon_data])

    otu_corrected_pd = pd.concat(
        [
            data[sample_label],
            data[batch_label],
            data[bio_label],
            pd.DataFrame(
                np_recon_data,
                index=data.index,
                columns=data.select_dtypes("number").columns,
            ),
        ],
        axis=1,
    )
    return otu_corrected_pd


def new_pre_train_abaco(
    vae,
    vae_optim_pre,
    discriminator,
    disc_optim,
    adv_optim,
    data_loader,
    normal_epochs,
    device,
    mog_epochs=0,
    w_contra=1.0,
    temp=0.1,
    w_elbo=1.0,
    w_disc=1.0,
    w_adv=1.0,
    disc_loss_type="CrossEntropy",
    n_disc_updates=1,
    label_smooth=0.1,
    normal=False,
    count=True,
):
    """
    PART OF THE NEW ABACO RUN WITH THREE PHASES. TWO OF THEM ARE COMPUTED HERE.
    Pre-training of conditional VAE with contrastive loss and adversarial mixing in latent space.
    """
    vae.train()
    discriminator.train()
    contra_criterion = SupervisedContrastiveLoss(temp)

    # FIRST STEP: PRE-TRAIN ABACO USING BETA = 0 FOR THE PRIOR

    total_steps = len(data_loader) * normal_epochs
    progress_bar = tqdm(
        range(total_steps),
        desc="Pre-training: VAE with contrastive learned embeddings by biological group",
    )

    for epoch in range(normal_epochs):

        for x, y_onehot, z_onehot in data_loader:
            # Move all tensors to the correct device
            x = x.to(device)
            if count == False:
                x_sum = x.sum(dim=1, keepdim=True)
                x = x / x_sum

            y_onehot = y_onehot.to(device)  # Batch one hot label
            z_onehot = z_onehot.to(device)  # Bio one hot label
            y_idx = y_onehot.argmax(1)
            z_idx = z_onehot.argmax(1)

            # === Step 1: Discriminator on latent (freeze encoder) ===
            with torch.no_grad():
                if normal == False:
                    pi, mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                    mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)
                    d_input = torch.cat([mu_bar, z_onehot], dim=1)

                else:
                    mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                    d_input = torch.cat([mu, z_onehot], dim=1)

            for _ in range(n_disc_updates):
                disc_optim.zero_grad()
                logits = discriminator(d_input)
                loss_disc = w_disc * F.cross_entropy(
                    logits, y_idx, label_smoothing=label_smooth
                )
                loss_disc.backward()
                disc_optim.step()

            # === Step 2: Adversarial update on encoder ===
            adv_optim.zero_grad()
            if normal == False:
                pi, mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)
                logits_fake = discriminator(torch.cat([mu_bar, z_onehot], dim=1))
            else:
                mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                logits_fake = discriminator(torch.cat([mu, z_onehot], dim=1))
            loss_adv = w_adv * adversarial_loss(
                pred_logits=logits_fake, true_labels=y_idx, loss_type=disc_loss_type
            )
            loss_adv.backward()
            adv_optim.step()

            # === Step 3: VAE reconstruction + contrastive ===
            vae_optim_pre.zero_grad()
            q_zx = vae.encoder(torch.cat([x, y_onehot], dim=1))
            latent = q_zx.rsample()
            p_xz = vae.decoder(torch.cat([latent, y_onehot], dim=1))

            recon_term = p_xz.log_prob(x).mean()
            # kl_beta = vae.beta * max(0.0, (epoch / normal_epochs))
            # kl_term = kl_beta * (q_zx.log_prob(latent) - vae.log_prob(latent)).mean()
            elbo_loss = -(recon_term)  # - kl_term)
            contra_loss = w_contra * contra_criterion(latent, z_idx)

            total_loss = w_elbo * elbo_loss + contra_loss
            total_loss.backward()
            vae_optim_pre.step()

            # Update progress bar
            progress_bar.set_postfix(
                elbo=f"{elbo_loss.item():.4f}",
                contra=f"{contra_loss.item():.4f}",
                disc=f"{loss_disc.item():.4f}",
                adv=f"{loss_adv.item():.4f}",
                epoch=f"{epoch}/{normal_epochs+1}",
            )
            progress_bar.update()

    progress_bar.close()

    # SECOND STEP: PRE-TRAIN ABACO WITH PRIOR DISTRIBUTION WITHOUT CONTRASTIVE LOSS

    total_steps = len(data_loader) * mog_epochs
    progress_bar = tqdm(
        range(total_steps),
        desc="Pre-training: VAE with prior distribution activated",
    )

    for epoch in range(mog_epochs):

        for x, y_onehot, z_onehot in data_loader:
            # Move all tensors to the correct device
            x = x.to(device)
            if count == False:
                x_sum = x.sum(dim=1, keepdim=True)
                x = x / x_sum

            y_onehot = y_onehot.to(device)  # Batch one hot label
            z_onehot = z_onehot.to(device)  # Bio one hot label
            y_idx = y_onehot.argmax(1)
            z_idx = z_onehot.argmax(1)

            # === Step 1: Discriminator on latent (freeze encoder) ===
            with torch.no_grad():
                if normal == False:
                    pi, mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                    mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)
                    d_input = torch.cat([mu_bar, z_onehot], dim=1)

                else:
                    mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                    d_input = torch.cat([mu, z_onehot], dim=1)

            for _ in range(n_disc_updates):
                disc_optim.zero_grad()
                logits = discriminator(d_input)
                loss_disc = w_disc * F.cross_entropy(
                    logits, y_idx, label_smoothing=label_smooth
                )
                loss_disc.backward()
                disc_optim.step()

            # === Step 2: Adversarial update on encoder ===
            adv_optim.zero_grad()
            if normal == False:
                pi, mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                mu_bar = (mu * pi.unsqueeze(2)).sum(dim=1)
                logits_fake = discriminator(torch.cat([mu_bar, z_onehot], dim=1))
            else:
                mu, _ = vae.encoder.encode(torch.cat([x, y_onehot], dim=1))
                logits_fake = discriminator(torch.cat([mu, z_onehot], dim=1))
            loss_adv = w_adv * adversarial_loss(
                pred_logits=logits_fake, true_labels=y_idx, loss_type=disc_loss_type
            )
            loss_adv.backward()
            adv_optim.step()

            # === Step 3: VAE reconstruction + contrastive ===
            vae_optim_pre.zero_grad()
            q_zx = vae.encoder(torch.cat([x, y_onehot], dim=1))
            latent = q_zx.rsample()
            p_xz = vae.decoder(torch.cat([latent, y_onehot], dim=1))

            recon_term = p_xz.log_prob(x).mean()
            kl_beta = vae.beta * max(0.0, (epoch / normal_epochs))
            kl_term = kl_beta * (q_zx.log_prob(latent) - vae.log_prob(latent)).mean()
            elbo_loss = -(recon_term - kl_term)
            # contra_loss = w_contra * contra_criterion(latent, z_idx)

            total_loss = w_elbo * elbo_loss  # + contra_loss
            total_loss.backward()
            vae_optim_pre.step()

            # Update progress bar
            progress_bar.set_postfix(
                elbo=f"{elbo_loss.item():.4f}",
                #                 contra=f"{contra_loss.item():.4f}",
                disc=f"{loss_disc.item():.4f}",
                adv=f"{loss_adv.item():.4f}",
                epoch=f"{epoch}/{mog_epochs+1}",
            )
            progress_bar.update()

    progress_bar.close()
