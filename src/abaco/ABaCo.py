import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import math
from abaco.dataloader import one_hot_encoding
import random
import seaborn as sns

# ---------- PRIOR CLASSES DEFINITIONS ---------- #


class NormalPrior(nn.Module):
    def __init__(self, d_z):
        """
        Define a Gaussian Normal prior distribution with zero mean and unit variance (Standard distribution).

        Parameters
        ----------
            d_z: int
                Dimension of the latent space
        """
        super().__init__()
        self.d_z = d_z
        self.mu = nn.Parameter(torch.zeros(self.d_z), requires_grad=False)
        self.var = nn.Parameter(torch.ones(self.d_z), requires_grad=False)

    def forward(self):
        """
        Return prior distribution. This allows the computation of KL-divergence by calling self.prior() in the VAE class.

        Returns
        -------
            prior: torch.distributions.Distribution
        """
        return td.Independent(td.Normal(loc=self.mu, scale=self.var), 1)


class MoGPrior(nn.Module):
    def __init__(self, d_z, n_comp, multiplier=1.0):
        """
        Define a Mixture of Gaussian Normal prior distribution with trainable mean and variance.

        Parameters
        ----------
            d_z: int
                Dimension of the latent space
            n_comp: int
                Number of components for the MoG distribution
            multiplier: float
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

        Returns
        -------
            prior: torch.distributions.Distribution
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

        Parameters
        ----------
            encoder_net: torch.nn.Module
                The encoder network, takes a tensor of dimension (batch, features) and outputs a tensor of dimension (batch, 2*d_z), where d_z is the dimension of the latent space.
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

        Parameters
        ----------
            x: torch.Tensor
                Tensor to be encoded

        Returns
        -------
            torch.distribution
                Gaussian Normal distribution use to calculate the posterior distribution with .rsample()
        """
        mu, var = self.encode(x)
        std = torch.sqrt(torch.nn.functional.softplus(var) + 1e-8)

        return td.Independent(td.Normal(loc=mu, scale=std), 1)

    def det_encode(self, x):
        """
        Computes the encoded point without stochastic component.

        Parameters
        ----------
            x: torch.Tensor
                Tensor to be encoded

        Returns
        -------
            mu: torch.Tensor
                Gaussian Noraml mean of the encoded point
        """
        mu, _ = self.encode(x)

        return mu

    def monte_carlo_encode(self, x, K=100):
        """
        Computes a Monte Carlo simulation of the same point to approximate the true posterior distribution.

        Parameters
        ----------
            x: torch.Tensor
                Tensor to be encoded
            K: int
                Number of Monte Carlo iterations

        Returns
        -------
            sample: torch.Tensor
                Mean from all samples (i.e. expectation of the encoded point estimated through Monte Carlo simulations)
        """
        mu, var = self.encode(x)
        std = torch.sqrt(torch.nn.functional.softplus(var) + 1e-8)

        dist = td.Independent(td.Normal(loc=mu, scale=std), 1)

        samples = dist.sample(sample_shape=(K,))

        sample = samples.mean(dim=0)

        return sample


class MoGEncoder(nn.Module):
    def __init__(self, encoder_net, n_comp):
        """
        Define a Mixture of Gaussians encoder to obtain the parameters of the MoG distribution.

        Parameters
        ----------
            encoder_net: torch.nn.Module
                The encoder network, takes a tensor of dimension (batch, features) and outputs a tensor of dimension (batch, n_comp*(2*d_z + 1)), where d_z is the dimension of the latent space, and n_comp the number of components of the MoG distribution.
            n_comp: int
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

        Parameters
        ----------
            x: torch.Tensor

        Returns
        -------
            mog_dist: MixtureOfGaussians
                MoG distribution use to calculate the approximate posterior with rsample()
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

        Parameters
        ----------
            x: torch.Tensor
                Tensor to be encoded

        Returns
        -------
            z: torch.Tensor
                Sum of MoG components means with proportion of the mixing probabilities of the encoded point
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
        Computes a Monte Carlo simulation of the same point to approximate the true posterior distribution.

        Parameters
        ----------
            x: torch.Tensor
                Tensor to be encoded
            K: int
                Number of Monte Carlo iterations

        Returns
        -------
            sample: torch.Tensor
                Mean from all samples (i.e. expectation of the encoded point estimated through Monte Carlo simulations)
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

        sample = samples.mean(dim=0)

        return sample


# ---------- DECODER CLASSES DEFINITIONS ---------- #


class NBDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Negative Binomial decoder to obtain the parameters of the distribution.

        Parameters
        ----------
            decoder_net: torch.nn.Module
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs a tensor of dimension (batch, 2*features), where d_z is the dimension of the latent space.
        """
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Computes the Negative Binomial distribution over the data space. What we are getting is the mean and the dispersion parameters, so it is needed a parameterization in order to get the NB distribution parameters: total_count (dispersion) and probs (dispersion/(dispersion + mean)).

        Parameters
        ----------
            z: torch.Tensor
                Latent space embeddings

        Returns
        -------
            torch.distribution
                Negative Binomial distribution use to calculate the likelihood using log_prob() method
        """
        mu, theta = torch.chunk(self.decoder_net(z), 2, dim=-1)
        # Ensure mean and dispersion are positive numbers and pi is in range [0,1]

        mu = F.softplus(mu)
        theta = F.softplus(theta) + 1e-4

        # Parameterization into NB parameters
        p = theta / (theta + mu)

        r = theta

        # Clamp values to avoid huge / small probabilities
        p = torch.clamp(p, min=1e-5, max=1 - 1e-5)

        # Create Negative Binomial component
        nb = td.NegativeBinomial(total_count=r, probs=p)
        # nb = td.Independent(nb, 1)

        return td.Independent(nb, 1)


class ZINBDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Zero-inflated Negative Binomial decoder to obtain the parameters of the distribution.

        Parameters
        ----------
            decoder_net: torch.nn.Module
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs a tensor of dimension (batch, 3*features), where d_z is the dimension of the latent space.
        """
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Computes the Zero-inflated Negative Binomial distribution over the data space. What we are getting is the zero probability, mean and the dispersion parameters, so it is needed a parameterization in order to get the NB distribution parameters: total_count (dispersion) and probs (dispersion/(dispersion + mean)).

        Parameters
        ----------
            z: torch.Tensor
                Latent space embeddings

        Returns
        -------
            torch.distribution
                Zero-inflated Negative Binomial distribution use to calculate the likelihood using log_prob() method
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


class DMDecoder(nn.Module):
    def __init__(self, decoder_net, total_count, eps=1e-8):
        """
        Define a Dirichlet-Multinomial decoder to obtain the parameters of the distribution.

        Parameters
        ----------
            decoder_net: torch.nn.Module
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs a tensor of dimension (batch, features), where d_z is the dimension of the latent space.
            total_count: int
                Total number of reads (or organisms) per sample. In practice, it is just x_i.sum(), where x_i is the sample i from the dataset.
            eps: float
                Small offset to avoid log(0) in probability computation.
        """
        super().__init__()
        self.decoder_net = decoder_net
        self.total_count = total_count
        self.eps = eps

    def forward(self, z):
        """
        Computes the Dirichlet-Multinomial distribution over the data space. We are obtaining the concentration parameter of the distribution, hence the decoder output would have shape (batch, features).

        Parameters
        ----------
            z: torch.Tensor
                Latent space embeddings

        Returns
        -------
            torch.distribution
                Dirichlet-Multinomial distribution use to calculate the likelihood using log_prob() method
        """
        conc_logits = self.decoder_net(z)
        concentration = F.softplus(conc_logits) + self.eps

        dm = DirichletMultinomial(
            total_count=self.total_count, concentration=concentration
        )

        return td.Independent(dm, 1)


class ZIDMDecoder(nn.Module):
    def __init__(self, decoder_net, total_count, eps=1e-8):
        """
        Define a Zero-inflated Dirichlet-Multinomial decoder to obtain the parameters of the distribution.

        Parameters
        ----------
            decoder_net: torch.nn.Module
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs a tensor of dimension (batch, 2*features), where d_z is the dimension of the latent space.
            total_count: int
                Total number of reads (or organisms) per sample. In practice, it is just x_i.sum(), where x_i is the sample i from the dataset.
            eps: float
                Small offset to avoid log(0) in probability computation.
        """
        super().__init__()
        self.decoder_net = decoder_net
        self.total_count = total_count
        self.eps = eps

    def forward(self, z):
        """
        Computes the Zero-inflated Dirichlet-Multinomial distribution over the data space. We are obtaining the zero probability and concentration parameter of the distribution, hence the decoder output would have shape (batch, 2*features).

        Parameters
        ----------
            z: torch.Tensor
                Latent space embeddings

        Returns
        -------
            torch.distribution
                Zero-inflated Dirichlet-Multinomial distribution use to calculate the likelihood using log_prob() method
        """
        conc_logits, pi_logits = torch.chunk(self.decoder_net(z), 2, dim=-1)

        concentration = F.softplus(conc_logits) + self.eps
        dm = DirichletMultinomial(
            total_count=self.total_count, concentration=concentration
        )

        pi = torch.sigmoid(pi_logits)
        zidm = ZIDM(dm=dm, pi=pi, eps=self.eps)

        return td.Independent(zidm, 1)


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
        Parameters
        ----------
            nb: torch.distributions.Independent(torch.distributions.NegativeBinomial)
                Negative Binomial distribution component
            pi: torch.Tensor
                Zero-inflation probability from decoder output
        """
        self.nb = nb
        self.pi = pi
        batch_shape = nb.batch_shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, x):
        """
        Defines the log_prob() function inherent from torch.distributions.Distribution.

        Parameters
        ----------
            x: torch.Tensor
                Observation used to calculate the log probability of the distribution fitting it.

        Returns
        -------
            log_p: torch.Tensor
                Log probability of the model fitting the observation.
        """
        nb_log_prob = self.nb.log_prob(x)  # log probability of NB where x > 0
        nb_prob_zero = torch.exp(
            self.nb.log_prob(torch.zeros_like(x))
        )  # probability of NB where x = 0
        log_prob_zero = torch.log(self.pi + (1 - self.pi) * nb_prob_zero + 1e-8)
        log_prob_nonzero = torch.log(1 - self.pi + 1e-8) + nb_log_prob

        log_p = torch.where(x == 0, log_prob_zero, log_prob_nonzero)

        return log_p

    def sample(self, sample_shape=torch.Size()):
        """
        Defines the sample() function, which is used to sample data points using the distribution parameters.
            For x = 0:
                Binary mask for zero-inflated values
            For x > 0:
                Sample from Negative Binomial distribution

        Parameters
        ----------
            sample_shape:
                Amount of samples. If not given sample() method would sampled an observation from the distribution.

        Returns
        -------
            zinb_sample:
                Sample(s) from the Zero-inflated Negative Binomial distribution.
        """
        shape = self._extended_shape(sample_shape)
        zero_inflated = torch.bernoulli(self.pi.expand(shape))

        nb_sample = self.nb.sample(sample_shape)

        zinb_sample = torch.where(
            zero_inflated.bool(), torch.zeros_like(nb_sample), nb_sample
        )

        return zinb_sample


@td.kl.register_kl(ZINB, ZINB)
def kl_zinb_zinb(p, q):
    """
    Approximation of the KL-divergence among two different Zero-inflated Negative Binomial distributions.

    Parameters
    ----------
        p: ZINB
            Zero-inflated negative binomial distribution 1
        q: ZINB
            Zero-inflated negative binomial distribution 2

    Returns
    -------
        kl: torch.Tensor
            KL-divergence between p and q
    """
    # Monte Carlo sampling from p
    num_samples = 5000
    samples = p.sample((num_samples,))
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)

    kl = (log_p - log_q).mean(dim=0)
    return kl


class DirichletMultinomial(td.Distribution):
    """
    Dirichlet-Multinomial distribution, defined by:
        p ~ Dirichlet(alpha)
        x | p ~ Multinomial(total_count, p)
    """

    arg_constraints = {
        "total_count": td.constraints.nonnegative_integer,
        "concentration": td.constraints.positive,
    }
    support = td.constraints.nonnegative_integer
    has_rsample = False

    def __init__(
        self, total_count: int, concentration: torch.Tensor, validate_args=None
    ):
        """
        Parameters
        ----------
            total_count: int
                Scalar integer for the Multinomial total counts N
            concentration: torch.Tensor
                Tensor of shape (..., num_categories) for Dirichlet alphas
        """
        self.total_count = total_count
        self.concentration = concentration
        batch_shape = concentration.shape[:-1]
        event_shape = concentration.shape[-1:]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def log_prob(self, x: torch.Tensor):
        """
        Defines the log_prob() function inherent from torch.distributions.Distribution.

        Parameters
        ----------
            x: torch.Tensor
                Observation used to calculate the log probability of the distribution fitting it

        Returns
        -------
            log_p: torch.Tensor
                Log probability of the distribution fitting the observation.
        """
        # if self._validate_args:
        #     total = x.sum(dim=-1)
        #     if not torch.all(total == self.total_count):
        #         raise ValueError("DirichletMultinomial counts must sum to total_count")

        alpha = self.concentration
        N = self.total_count

        term1 = torch.lgamma(
            torch.tensor(N + 1, dtype=torch.float, device=x.device)
        ) - torch.lgamma(x + 1).sum(dim=-1)

        sum_alpha = alpha.sum(dim=-1)
        term2 = torch.lgamma(sum_alpha) - torch.lgamma(N + sum_alpha)

        term3 = torch.lgamma(x + alpha).sum(dim=-1) - torch.lgamma(alpha).sum(dim=-1)

        log_p = term1 + term2 + term3

        return log_p

    def sample(self, sample_shape=torch.Size()):
        """
        Defines the sample() function, which is used to sample data points using the distribution parameters.

        Parameters
        ----------
            sample_shape:
                Amount of samples. If not given sample() method would sampled an observation from the distribution.

        Returns
        -------
            dm_sample:
                Sample(s) from the Dirichlet Multinomial distribution.
        """
        # shape = self._extended_shape(sample_shape)
        p = td.Dirichlet(self.concentration).sample(sample_shape)

        batch_dims = p.shape[:-1]
        C = p.size(-1)

        # flatten batch dims
        p_flat = p.reshape(-1, C)

        # total_count per flat sample
        tc = self.total_count

        if isinstance(tc, torch.Tensor):
            tc_flat = tc.reshape(-1)

        else:
            tc_flat = None

        counts = []
        for i, probs in enumerate(p_flat):
            n = int(tc_flat[i]) if tc_flat is not None else self.total_count
            idx = torch.multinomial(probs, n, replacement=True)
            counts.append(torch.bincount(idx, minlength=C))

        x = torch.stack(counts, dim=0)

        dm_sample = x.reshape(*batch_dims, C)

        return dm_sample


class ZIDM(td.Distribution):
    """
    Zero-inflated Dirichlet-Multinomial (ZIDM) distribution, mixture of structural zeros per-category with a Dirichlet-Multinomial core.
    """

    arg_constraints = {}
    support = td.constraints.nonnegative_integer
    has_rsample = False

    def __init__(
        self,
        dm: DirichletMultinomial,
        pi: torch.Tensor,
        eps: float = 1e-8,
        validate_args=None,
    ):
        """
        Parameters
        ----------
            dm: DirichletMultinomial
                    Dirichlet Multinomial instance of the distribution
            pi: torch.Tensor
                Tensor of shape (..., num_categories), zero-inflation probability per category
            eps: float
                Small offset to ensure non-zero concentration for masked-out categories
        """
        self.dm = dm
        self.pi = pi
        self.eps = eps
        batch_shape = dm.batch_shape
        event_shape = dm.event_shape
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def log_prob(self, x: torch.Tensor):
        """
        Defines the log_prob() function inherent from torch.distributions.Distribution.

        Parameters
        ----------
            x: torch.Tensor
                Observation used to calculate the log probability of the distribution fitting it

        Returns
        -------
            log_zidm: torch.Tensor
                Log probability of the distribution fitting the observation
        """
        mask_nonzero = (x > 0).float()
        term_pi = torch.log((1 - self.pi) + self.eps) * mask_nonzero
        log_dm = self.dm.log_prob(x)

        log_zidm = term_pi.sum(dim=-1) + log_dm

        return log_zidm

    def sample(self, sample_shape=torch.Size()):
        """
        Defines the sample() function, which is used to sample data points using the distribution parameters.

        Parameters
        ----------
            sample_shape:
                Amount of samples. If not given sample() method would sampled an observation from the distribution.

        Returns
        -------
            zinb_sample:
                Sample(s) from the Zero-inflated Dirichlet Multinomial distribution.
        """
        shape = self._extended_shape(sample_shape)
        zero_mask = torch.bernoulli(self.pi.expand(shape))
        alpha_adj = self.dm.concentration * (1 - zero_mask) + self.eps
        dm_adj = DirichletMultinomial(
            total_count=self.dm.total_count, concentration=alpha_adj
        )
        sample = dm_adj.sample()
        return sample


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

        Parameters
        ----------
            sample_shape:
                Amount of samples. If not given it would return a sample

        Returns
        -------
            sample: torch.Tensor
                Sample from the MoG distribution
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

        Parameters
        ----------
            value: torch.Tensor
                Value to be used to calculate the MoG probability of fitting it

        Returns
        -------
            log_prob: torch.Tensor
                Log probability of the distribution fitting the value
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
    """
    Approximation of the KL-divergence among two different Mixture of Gaussians distributions.

    Parameters
    ----------
        p: MixtureOfGaussians
            MoG distribution 1
        q: MixtureOfGaussians
            MoG distribution 2

    Returns
    -------
        kl: torch.Tensor
            KL-divergence between p and q
    """
    # Monte Carlo sampling from p
    num_samples = 5000
    samples = p.rsample((num_samples,))
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)

    kl = (log_p - log_q).mean(dim=0)
    return kl


# ---------- VAE CLASSES DEFINITIONS ---------- #


# class VAE(nn.Module):
#     """
#     Define a Variational Autoencoder (VAE) model.
#     """

#     def __init__(self, prior, decoder, encoder, beta=1.0):
#         """
#         Parameters:
#         prior: [torch.nn.Module]
#            The prior distribution over the latent space.
#         decoder: [torch.nn.Module]
#               The decoder distribution over the data space.
#         encoder: [torch.nn.Module]
#                 The encoder distribution over the latent space.
#         """

#         super(VAE, self).__init__()
#         self.prior = prior
#         self.decoder = decoder
#         self.encoder = encoder
#         self.beta = beta

#     def elbo(self, x):
#         """
#         Compute the ELBO for the given batch of data.

#         Parameters:
#         x: [torch.Tensor]
#            A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
#            n_samples: [int]
#            Number of samples to use for the Monte Carlo estimate of the ELBO.
#         """
#         q = self.encoder(x)
#         z = q.rsample()
#         elbo = torch.mean(
#             self.decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()),
#             dim=0,
#         )

#         return elbo

#     def kl_div_loss(self, x):
#         q = self.encoder(x)
#         z = q.rsample()
#         kl_loss = torch.mean(
#             self.beta * td.kl_divergence(q, self.prior()),
#             dim=0,
#         )
#         return kl_loss

#     def sample(self, n_samples=1):
#         """
#         Sample from the model.

#         Parameters:
#         n_samples: [int]
#            Number of samples to generate.
#         """
#         z = self.prior().sample(torch.Size([n_samples]))
#         return self.decoder(z).sample()

#     def forward(self, x):
#         """
#         Compute the negative ELBO for the given batch of data.

#         Parameters:
#         x: [torch.Tensor]
#            A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
#         """
#         return -self.elbo(x)

#     def get_posterior(self, x):
#         """
#         Given a set of points, compute the posterior distribution.

#         Parameters:
#         x: [torch.Tensor]
#             Samples to pass to the encoder
#         """
#         q = self.encoder(x)
#         z = q.rsample()
#         return z

#     def pca_posterior(self, x):
#         """
#         Given a set of points, compute the PCA of the posterior distribution.

#         Parameters:
#         x: [torch.Tensor]
#             Samples to pass to the encoder
#         """
#         z = self.get_posterior(x)
#         pca = PCA(n_components=2)
#         return pca.fit_transform(z.detach().cpu())

#     def pca_prior(self, n_samples):
#         """
#         Given a number of samples, get the PCA from the sampling of the prior distribution.
#         """
#         samples = self.prior().sample(torch.Size([n_samples]))
#         pca = PCA(n_components=2)
#         return pca.fit_transform(samples.detach().cpu())


class ConditionalVAE(nn.Module):

    def __init__(self, prior, decoder, encoder, beta=1.0):
        """
        Define a conditional Variational Autoencoder (VAE) model.

        Parameters
        ----------
            prior: torch.nn.Module
                The prior distribution over the latent space.
            decoder: torch.nn.Module
                The decoder distribution over the data space.
            encoder: torch.nn.Module
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

        Parameters
        ----------
            x: torch.Tensor
                A tensor of dimension (batch_size, feature_dim1, feature_dim2, ...)
            n_samples: int
                Number of samples to use for the Monte Carlo estimate of the ELBO

        Returns
        -------
            elbo: torch.Tensor
                Evidence Lower Bound
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()),
            dim=0,
        )

        return elbo

    def kl_div_loss(self, x):
        """
        KL-divergence between the prior and the posterior distribution.

        Parameters
        ----------
            x: torch.Tensor
                Observation to be encoded in order to get the approximate posterior distribution

        Returns
        -------
            kl_loss: torch.Tensor
                KL-divergence loss
        """
        q = self.encoder(x)
        # z = q.rsample()
        kl_loss = torch.mean(
            self.beta * td.kl_divergence(q, self.prior()),
            dim=0,
        )
        return kl_loss

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters
        ----------
            n_samples: int
                Number of samples to generate

        Returns
        -------
            samples: torch.Tensor
                Samples generated from the model
        """
        z = self.prior().sample(torch.Size([n_samples]))

        samples = self.decoder(z).sample()
        return samples

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters
        ----------
            x: torch.Tensor
                A tensor of dimension (batch_size, feature_dim1, feature_dim2)

        Returns
        -------
            loss: torch.Tensor
                Negative ELBO
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

        loss = -recon_term + kl_term

        return loss

    def get_posterior(self, x):
        """
        Given a set of points, compute the samples of the posterior distribution.

        Parameters
        ----------
            x: torch.Tensor
                Samples to pass to the encoder to obtain the posterior distribution from

        Returns
        -------
            z: torch.Tensor
                Encoded points sampled from the posterior distribution
        """
        q = self.encoder(x)
        z = q.rsample()
        return z

    def log_prob(self, z):
        return self.prior().log_prob(z)

    def pca_posterior(self, x):
        """
        Given a set of points, compute the PCA of the posterior distribution.

        Parameters
        ----------
            x: torch.Tensor
                Samples to pass to the encoder
        """
        z = self.get_posterior(x)
        pca = PCA(n_components=2)
        return pca.fit_transform(z.detach().cpu())

    def pca_prior(self, n_samples):
        """
        Given a number of samples, get the PCA from the sampling of the prior distribution.

        Parameters
        ----------
            n_samples: int
                Number of samples from the prior distribution
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
        # z = q.rsample()
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

        for _decoder in self.decoders:
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
    Define a VampPrior Variational Autoencoder model. Class is used for the baseline application of ABaCo model.
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
        """
        Parameters
        ----------
            encoder: torch.nn.Module
                Encoder class of the VAE
            decoder: torch.nn.Module
                Decoder class of the VAE
            input_dim: int
                Dimension of the input
            batch_dim: int
                Number of batches in the data
            n_comps: int
                Number of components of the VampPrior Mixture Model
            d_z: int
                Latent space dimension
            beta: int
                Weight for the KL-divergence term of the ELBO
            data_loader: torch.utils.data.DataLoader
                DataLoader class with the training data
        """
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
        for _decoder in self.decoders:
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
        """
        Parameters
        ----------
            net: torch.nn.Module
                Feed-forward neural network for the Batch Discriminator.
        """
        super().__init__()
        self.net = net

    def forward(self, x):
        """
        Computes the forward pass through the discriminator.

        Parameters
        ----------
            x: torch.Tensor
                Input to be passed through the model

        Returns
        -------
            batch_class: torch.Tensor
                Prediction of the batch of origin of the observation
        """
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
            if not count:
                x_sum = x.sum(dim=1, keepdim=True)
                x = x / x_sum

            y_onehot = y_onehot.to(device)  # Batch one hot label
            z_onehot = z_onehot.to(device)  # Bio one hot label
            y_idx = y_onehot.argmax(1)
            z_idx = z_onehot.argmax(1)

            # === Step 1: Discriminator on latent (freeze encoder) ===
            with torch.no_grad():
                if not normal:
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
            if not normal:
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
                epoch=f"{epoch}/{epochs + 1}",
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
            # z = loader_data[2].to(device).float()  # Bio type label

            # VAE ELBO computation with masked batch label
            vae_optim_post.zero_grad()

            # Forward pass to encoder
            q_zx = vae.encoder(torch.cat([x, y], dim=1))

            # Sample from encoded point
            latent_points = q_zx.rsample()

            # Forward pass to the decoder
            p_xz = vae.decoder(torch.cat([latent_points, alpha * y], dim=1))

            # Log probabilities of prior and posterior
            # log_q_zx = q_zx.log_prob(latent_points)
            # log_p_z = vae.log_prob(latent_points)

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
                epoch=f"{epoch + 1}/{epochs}",
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
    pre_epochs: int = 2000,
    post_epochs: int = 2000,
    kl_cycle: bool = True,
    smooth_annealing: bool = False,
    # VAE Model architecture
    encoder_net: list = [1024, 512, 256],
    decoder_net: list = [256, 512, 1024],
    vae_act_func=nn.ReLU(),
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
    adv_lr: float = 1e-5,
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
            raise NotImplementedError(
                "Relative abundance data type isn't implemented yet to ABaCo. Set variable 'count' to True."
            )
            # decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            # decoder_net.append(2 * input_size)  # last layer
            # modules = []
            # for i in range(len(decoder_net) - 1):
            #     modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
            #     modules.append(vae_act_func)
            # modules.pop()  # Drop last activation function
            # decoder = ZIDirichletDecoder(nn.Sequential(*modules))

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
            raise NotImplementedError(
                "Relative abundance data type isn't implemented yet to ABaCo. Set variable 'count' to True."
            )
            # decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            # decoder_net.append(2 * input_size)  # last layer
            # modules = []
            # for i in range(len(decoder_net) - 1):
            #     modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
            #     modules.append(vae_act_func)
            # modules.pop()  # Drop last activation function
            # decoder = ZIDirichletDecoder(nn.Sequential(*modules))

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
            raise NotImplementedError(
                "Relative abundance data type isn't implemented yet to ABaCo. Set variable 'count' to True."
            )
            # decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            # decoder_net.append(2 * input_size)  # last layer
            # modules = []
            # for i in range(len(decoder_net) - 1):
            #     modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
            #     modules.append(vae_act_func)
            # modules.pop()  # Drop last activation function
            # decoder = ZIDirichletDecoder(nn.Sequential(*modules))

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
        raise ValueError(
            f"Prior distribution '{prior}' isn't a valid option: 'VMM', 'MoG' or 'Normal'"
        )

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
    if not new_pre_train:
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
    device: torch.device,
    data: pd.DataFrame,
    dataloader: torch.utils.data.DataLoader,
    sample_label: str,
    batch_label: str,
    bio_label,
    seed=42,
    det_encode=False,
    monte_carlo=100,
):
    """
    Function used to reconstruct data using trained ABaCo model.

    Parameters
    ----------
    model:
        Trained ABaCo model.
    device: torch.device
        Device to run the model on, e.g., "cuda" or "cpu".
    data: pd.DataFrame
        DataFrame containing the data to be reconstructed.
    dataloader: torch.utils.data.DataLoader
        Pytorch DataLoader for the data to be reconstructed.
    sample_label: str
        Column name in the DataFrame that contains unique ids for the observations/samples.
    batch_label: str
        Column name in the DataFrame that contains ids for
        the batch/factor groupings to be corrected by abaco. e.g. dates of sample analysis
    bio_label: str
        Column name in the DataFrame that contains biological groupings where there is the
        biological/experimental factor variation for abaco to retain when correcting batch effect
        e.g., experimental condition
    seed: int, optional
        Random seed for reproducibility. Default is 42.
    det_encode: bool, optional
        If True, use deterministic encoding. Default is False.
    monte_carlo: int, optional
        Number of Monte Carlo samples to use for reconstruction. Default is 100.
        Setting at 1 is the same as just sampling from the final ZINB distribution obtained from the trained model
    Returns
    -------
    otu_corrected_pd: pd.DataFrame
        DataFrame containing the reconstructed data with batch and biological labels.
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
        if det_encode:
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
            if not count:
                x_sum = x.sum(dim=1, keepdim=True)
                x = x / x_sum

            y_onehot = y_onehot.to(device)  # Batch one hot label
            z_onehot = z_onehot.to(device)  # Bio one hot label
            y_idx = y_onehot.argmax(1)
            z_idx = z_onehot.argmax(1)

            # === Step 1: Discriminator on latent (freeze encoder) ===
            with torch.no_grad():
                if not normal:
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
            if not normal:
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
                epoch=f"{epoch}/{epochs + 1}",
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
            # z = loader_data[2].to(device).float()  # Bio type label

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
            # log_q_zx = q_zx.log_prob(latent_points)
            # log_p_z = vae.log_prob(latent_points)

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
                epoch=f"{epoch + 1}/{epochs}",
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
            raise NotImplementedError(
                "Relative abundance data type isn't implemented yet to ABaCo. Set variable 'count' to True."
            )
            # for _ in range(n_dec):
            #     decoder_net = [
            #         d_z + n_batches
            #     ] + decoder_net  # first value: conditional
            #     decoder_net.append(2 * input_size)  # last layer
            #     modules = []
            #     for i in range(len(decoder_net) - 1):
            #         modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
            #         modules.append(vae_act_func)
            #     modules.pop()  # Drop last activation function
            #     decoder = ZIDirichletDecoder(nn.Sequential(*modules))
            #     decoders.append(decoder)

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
            raise NotImplementedError(
                "Relative abundance data type isn't implemented yet to ABaCo. Set variable 'count' to True."
            )
            # decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            # decoder_net.append(2 * input_size)  # last layer
            # modules = []
            # for i in range(len(decoder_net) - 1):
            #     modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
            #     modules.append(vae_act_func)
            # modules.pop()  # Drop last activation function
            # decoder = ZIDirichletDecoder(nn.Sequential(*modules))

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
            raise NotImplementedError(
                "Relative abundance data type isn't implemented yet to ABaCo. Set variable 'count' to True."
            )
            # decoder_net = [d_z + n_batches] + decoder_net  # first value: conditional
            # decoder_net.append(2 * input_size)  # last layer
            # modules = []
            # for i in range(len(decoder_net) - 1):
            #     modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
            #     modules.append(vae_act_func)
            # modules.pop()  # Drop last activation function
            # decoder = ZIDirichletDecoder(nn.Sequential(*modules))

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
        raise ValueError("Prior distribution select isn't a valid option.")

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
        if det_encode:
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
            if not count:
                x_sum = x.sum(dim=1, keepdim=True)
                x = x / x_sum

            y_onehot = y_onehot.to(device)  # Batch one hot label
            z_onehot = z_onehot.to(device)  # Bio one hot label
            y_idx = y_onehot.argmax(1)
            z_idx = z_onehot.argmax(1)

            # === Step 1: Discriminator on latent (freeze encoder) ===
            with torch.no_grad():
                if not normal:
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
            if not normal:
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
                epoch=f"{epoch}/{normal_epochs + 1}",
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
            if not count:
                x_sum = x.sum(dim=1, keepdim=True)
                x = x / x_sum

            y_onehot = y_onehot.to(device)  # Batch one hot label
            z_onehot = z_onehot.to(device)  # Bio one hot label
            y_idx = y_onehot.argmax(1)
            z_idx = z_onehot.argmax(1)

            # === Step 1: Discriminator on latent (freeze encoder) ===
            with torch.no_grad():
                if not normal:
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
            if not normal:
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
                epoch=f"{epoch}/{mog_epochs + 1}",
            )
            progress_bar.update()

    progress_bar.close()


# ----- metaABaCo function ----- #


class MoCPPrior(nn.Module):
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

    def forward(self, k_ohe):
        """
        Return prior distribution, allowing for the computation of the KL-divergence by calling self.prior().

        Returns:
            prior: [torch.distributions.Distribution]
        """
        # Get parameters for each MoG component
        mu_k = k_ohe @ self.mu
        std_k = k_ohe @ torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)

        return td.Independent(td.Normal(loc=mu_k, scale=std_k), 1)

    def cluster_loss(self):
        """
        Compute the clustering loss for the MoG prior. This loss encourages the components to be well separated
        by maximizing the pairwise KL divergence between the Gaussian components.
        """
        # Compute softplus to ensure positive variances
        stds = torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)
        mus = self.mu
        n = self.n_comp

        # Compute pairwise KL divergences between all components
        kl_matrix = torch.zeros((n, n), device=mus.device)
        for i in range(n):
            for j in range(n):
                if i != j:
                    mu1, mu2 = mus[i], mus[j]
                    std1, std2 = stds[i], stds[j]
                    var1, var2 = std1**2, std2**2

                    # KL(N1 || N2) for diagonal Gaussians
                    kl = 0.5 * (
                        torch.sum(var1 / var2)
                        + torch.sum((mu2 - mu1) ** 2 / var2)
                        - self.d_z
                        + torch.sum(torch.log(var2))
                        - torch.sum(torch.log(var1))
                    )
                    kl_matrix[i, j] = kl

        # Take the minimum KL divergence between any two components
        positive_kl = kl_matrix[kl_matrix > 0]
        if positive_kl.numel() == 0:
            # Handle the case where there are no positive KL values
            min_kl = 0.0  # or float('inf'), or another default
        else:
            min_kl = positive_kl.min()
        # Loss is inverse of min KL (maximize separation)
        return 1.0 / (min_kl + 1e-8)


class VMMPrior(nn.Module):
    def __init__(
        self, d_z, n_features, n_comp, n_batch, encoder, multiplier=1.0, dataloader=None
    ):
        """
        Define a VampPrior Mixture Model prior distribution.

        Parameters:
            d_z: [int]
                Dimension of the latent space
            n_u: [int]
                Number of pseudo-inputs for the VMM distribution
            multiplier: [float]
                Parameter that controls sparsity of each Gaussian component
        """
        super().__init__()
        self.d_z = d_z
        self.n_features = n_features
        self.n_comp = n_comp
        self.n_batch = n_batch
        self.encoder = encoder
        self.dataloader = dataloader

        if self.dataloader is not None:
            self.u = self.sample_from_dataloader()
        else:
            self.u = nn.Parameter(
                torch.cat(
                    [
                        torch.rand(n_comp, n_features) * multiplier,
                        torch.zeros(n_comp, n_batch),
                    ],
                    dim=1,
                )
            )

        self.var = nn.Parameter(torch.randn(n_comp, self.d_z))

    def sample_from_dataloader(self):
        all_data = []
        bio_label = []
        # Collect until we have at least K samples
        for batch in self.dataloader:
            x = batch[0]
            z = batch[2]  # biological variability
            all_data.append(x)
            bio_label.append(z)
            if len(all_data) * x.shape[0] >= self.n_comp:
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
        zeros_pad = torch.zeros(self.n_comp, self.n_batch, device=selected_u.device)
        selected_u = torch.cat([selected_u, zeros_pad], dim=1)

        # Return as a learnable parameter
        return nn.Parameter(selected_u.clone().detach().requires_grad_(True))

    def forward(self, k_ohe):
        """
        Return prior distribution, allowing for the computation of the KL-divergence by calling self.prior().

        Parameters:
            k_ohe: [torch.tensor]
                One-hot encoded tensor with the corresponding component the point belongs to.
            b_ohe: [torch.tensor]
                One-hot encoded tensor with the corresponding batch label, necessary to append at the Encoder input.
            encoder: [nn.Module]
                Encoder used for getting the centroid of the cluster by encoding the pseudo-input.

        Returns:
            prior: [torch.distributions.Distribution]
        """
        # Encode the pseudo-input
        u_k = k_ohe @ self.u
        _, mu_k, _ = self.encoder.encode(u_k)
        mu_k = mu_k[torch.arange(mu_k.size(0)), k_ohe.argmax(dim=1), :]  # (batch, d_z)

        # Get parameters for each MoG component
        std_k = k_ohe @ torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)

        return td.Independent(td.Normal(loc=mu_k, scale=std_k), 1)

    def cluster_loss(self):
        """
        Compute the clustering loss for the MoG prior. This loss encourages the components to be well separated
        by maximizing the pairwise KL divergence between the Gaussian components.
        """

        # Compute softplus to ensure positive variances
        stds = torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)
        _, mus, _ = self.encoder.encode(self.u)
        n = self.n_comp

        # Compute pairwise KL divergences between all components
        kl_matrix = torch.zeros((n, n), device=mus.device)
        for i in range(n):
            for j in range(n):
                if i != j:
                    mu1, mu2 = mus[i], mus[j]
                    std1, std2 = stds[i], stds[j]
                    var1, var2 = std1**2, std2**2

                    # KL(N1 || N2) for diagonal Gaussians
                    kl = 0.5 * (
                        torch.sum(var1 / var2)
                        + torch.sum((mu2 - mu1) ** 2 / var2)
                        - self.d_z
                        + torch.sum(torch.log(var2))
                        - torch.sum(torch.log(var1))
                    )
                    kl_matrix[i, j] = kl

        # Take the minimum KL divergence between any two components
        positive_kl = kl_matrix[kl_matrix > 0]
        if positive_kl.numel() == 0:
            # Handle the case where there are no positive KL values
            min_kl = 0.0  # or float('inf'), or another default
        else:
            min_kl = positive_kl.min()
        # Loss is inverse of min KL (maximize separation)
        return 1.0 / (min_kl + 1e-8)


class MoCPEncoder(nn.Module):
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
        Computes the Categorical distribution over the latent space, sampling the component index
        and returning the parameters of the selected component.

        Parameters:
            x: [torch.Tensor]
        """
        pis, means, stds = self.encode(x)

        # From the categorical distribution, sample the component index
        probs = F.softmax(pis, dim=-1)
        cat = td.Categorical(probs=probs)
        k = cat.sample()

        # Get the parameters of the selected component
        k_exp = k.view(-1, 1, 1).expand(-1, 1, means.size(-1))
        mu_k = means.gather(dim=1, index=k_exp).squeeze(1)
        std_k = stds.gather(dim=1, index=k_exp).squeeze(1)

        return td.Independent(td.Normal(loc=mu_k, scale=std_k), 1)


class VAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        prior,
        input_size,
        device,
    ):
        """
        Variational Autoencoder class definition.

        Parameters:
            encoder: [torch.nn.Module]
                Encoder network, takes a tensor of dimension (batch, features) and outputs a tensor of dimension (batch, 2*d_z)
            decoder: [torch.nn.Module]
                Decoder network, takes a tensor of dimension (batch, d_z) and outputs a tensor of dimension (batch, features)
            prior: [torch.distributions.Distribution]
                Prior distribution over the latent space
            input_size: [int]
                Dimension of the input data
            device: [str]
                Device to use for computations
            prior_type: [str]
                Type of prior distribution used in the VAE
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.input_size = input_size
        self.device = device

    def encode(self, x):
        """
        Forward pass through the VAE.

        Parameters:
            x: [torch.Tensor]
                Input data tensor of shape (batch, features)
        """
        z = self.encoder(x)

        return z

    def decode(self, z):
        """
        Decode the latent space representation to the data space.

        Parameters:
            z: [torch.Tensor]
                Latent space tensor of shape (batch, d_z)
        """
        recon_x = self.decoder(z)

        return recon_x

    def kl_divergence(self, z, k_ohe=None):
        """
        Compute the KL-divergence between the prior and the posterior distribution.

        Parameters:
            z: [torch.Tensor]
                Latent space tensor of shape (batch, d_z)
        """
        # 1. Encode the input data to get the posterior distribution parameters
        q_zx = self.encoder(z)

        # 2. Get the prior distribution
        if isinstance(self.prior, MoCPPrior):
            p_z = self.prior(k_ohe)  # select Gaussian component of the prior
        else:
            p_z = self.prior()  # sample from standard Gaussian prior

        # 3. Compute the KL-divergence
        kl_div = td.kl_divergence(q_zx, p_z)

        return kl_div

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        return q.rsample()

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


class metaABaCo(nn.Module):
    def __init__(
        self,
        data,
        n_bios,
        bio_label,
        n_batches,
        batch_label,
        n_features,
        device,
        # VAE parameters
        prior="MoG",
        pdist="ZINB",
        d_z=16,
        epochs=[1000, 2000, 2000],
        encoder_net=[512, 256, 128],
        decoder_net=[128, 256, 512],
        vae_act_fun=nn.ReLU(),
        # Discriminator parameters
        disc_net=[128, 64],
        disc_act_fun=nn.ReLU(),
    ):
        """
        Function to create the metaABaCo model.

        Parameters
        ----------
        data: pd.DataFrame
            Pre-processed DataFrame to correct. Only feature columns to correct should be of numerical data type.
        n_bios: int
            Number of labels or (potential) clusters based on biological variance. For example, if 2 experimental
            conditions (e.g., control and treatment) then n_bios = 2.
        bio_label: char
            Column label where biological labels are contained in data.
        n_batches: int
            Number of batches in the dataset. For example, if samples were sequenced in
            5 batches (e.g., 5 different dates) then batches = 5.
        batch_label: char
            Column label where batch labels are contained in data.
        n_features: int
            Number of features in the input data, columns. For example, if the input is a gene expression matrix with 1000 genes,
            then input_size = 1000.
        device: torch.device
            Device to run the model on, e.g., "cuda" or "cpu".
        prior: str
            Prior distribution used. Baseline is "VMM" (VampPrior Mixture Model). Options are "VMM" and
            "MoG" (Mixture of Gaussians).
        pdist: str
            Output distribution used. Baseline is "ZINB" (Zero-inflated Negative Binomial).
        d_z: int
            Dimensionality of the latent space. For example, if d_z = 16, then the latent space will have 16 dimensions.
        epochs: list
            Number of epochs for first, second and third phase of ABaCo. Default is [1000, 2000, 2000]
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
        disc_act_fun: nn.Module
            Activation function for the discriminator. Default is nn.ReLU().
        """
        super().__init__()

        # Define known model parameters
        self.device = device
        self.data = data
        self.n_bios = n_bios
        self.bio_label = bio_label
        self.n_batches = n_batches
        self.batch_label = batch_label
        self.n_features = n_features
        self.d_z = d_z
        self.phase_1_epochs = epochs[0]
        self.phase_2_epochs = epochs[1]
        self.phase_3_epochs = epochs[2]

        # Defince dataloader
        self.dataloader = DataLoader(
            TensorDataset(
                torch.tensor(
                    self.data.select_dtypes(include="number").values,
                    dtype=torch.float32,
                ),  # samples
                one_hot_encoding(self.data[self.batch_label])[
                    0
                ],  # one hot encoded batch information
                one_hot_encoding(self.data[self.bio_label])[
                    0
                ],  # one hot encoded biological information
            ),
            batch_size=len(self.data),
        )

        for x, _y, _z in self.dataloader:  # just one iteration
            self.total_count = x.sum(dim=1).to(self.device)

        # Define Encoder
        encoder_net = [n_features + n_batches] + encoder_net  # first layer: conditional
        encoder_net.append(
            n_bios * (2 * d_z + 1)
        )  # last layer: gaussian parameters (mu*d_z + sigma*d_z + pi = 2*d_z + 1) times the number of gaussians (n_bios)
        modules = []
        for i in range(len(encoder_net) - 1):
            modules.append(nn.Linear(encoder_net[i], encoder_net[i + 1]))
            modules.append(vae_act_fun)
        modules.pop()  # Drop last activation function

        if prior == "MoG":
            self.encoder = MoCPEncoder(nn.Sequential(*modules), n_bios)
            self.prior = MoCPPrior(d_z, n_bios)

        elif prior == "VMM":
            self.encoder = MoGEncoder(nn.Sequential(*modules), n_bios)
            self.prior = VMMPrior(
                d_z,
                n_features,
                n_bios,
                n_batches,
                self.encoder,
                dataloader=self.dataloader,
            )

        else:
            raise NotImplementedError(
                "Only 'MoG' and 'VMM' prior are currently implemented in metaAbaco."
            )

        # Define Decoder
        decoder_net = [d_z + n_batches] + decoder_net  # first layer: conditional

        if pdist == "ZINB":
            decoder_net.append(
                3 * n_features
            )  # last layer: ZINB distribution parameters (n_features * (dispersion + dropout + mean))

        elif pdist == "NB":
            decoder_net.append(
                2 * n_features
            )  # last layer: ZINB distribution parameters (n_features * (dispersion + mean))

        elif pdist == "DM":
            decoder_net.append(
                n_features
            )  # last layer: Dirichlet-Multinomial distribution parameters (n_feature * concentration)

        elif pdist == "ZIDM":
            decoder_net.append(
                2 * n_features
            )  # last layer: ZIDM distribution parameters (n_features * (concentration + dropout))

        else:
            raise NotImplementedError(
                "Only 'ZINB', 'DM' and 'ZIDM' decoders are currently implemented in metaAbaco."
            )

        modules = []
        for i in range(len(decoder_net) - 1):
            modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
            modules.append(vae_act_fun)
        modules.pop()  # Drop last activation function

        if pdist == "ZINB":
            self.decoder = ZINBDecoder(nn.Sequential(*modules))

        elif pdist == "NB":
            self.decoder = NBDecoder(nn.Sequential(*modules))

        elif pdist == "DM":
            self.decoder = DMDecoder(nn.Sequential(*modules), self.total_count)

        elif pdist == "ZIDM":
            self.decoder = ZIDMDecoder(nn.Sequential(*modules), self.total_count)

        else:
            raise NotImplementedError(
                "Only 'NB', 'ZINB', 'DM', and 'ZIDM' decoders are currently implemented in metaAbaco."
            )

        # Define the VAE
        self.vae = VAE(self.encoder, self.decoder, self.prior, n_features, device).to(
            device
        )

        # Define Batch Discriminator
        disc_net = [d_z + n_bios] + disc_net  # first layer: conditional
        disc_net.append(n_batches)  # last layer
        modules = []
        for i in range(len(disc_net) - 1):
            modules.append(nn.Linear(disc_net[i], disc_net[i + 1]))
            modules.append(disc_act_fun)
        modules.pop()  # remove last activation function

        self.disc = BatchDiscriminator(nn.Sequential(*modules)).to(device)

    def train_vae(
        self,
        train_loader,
        optimizer,
        w_elbo_nll=1.0,
        w_elbo_kl=1.0,
        w_bio_penalty=1.0,
        w_cluster_penalty=1.0,
    ):
        """
        Train the conditional VAE model. If clustering prior is used, penalization term is applied to increase sparsity of the clusters.

        Parameters:
            vae: [VAE]
                Variational Autoencoder model
            train_loader: [torch.utils.data.DataLoader]
                DataLoader for the training data
            optimizer: [torch.optim.Optimizer]
                Optimizer for training
            epochs: [int]
                Number of training epochs
            device: [str]
                Device to use for computations
        """
        self.vae.train()

        total_steps = len(train_loader) * self.phase_1_epochs
        progress_bar = tqdm(
            range(total_steps),
            desc="Training: VAE for learning meaningful embeddings",
        )

        for epoch in range(self.phase_1_epochs):
            total_loss = 0.0
            data_iter = iter(train_loader)
            for loader_data in data_iter:
                x = loader_data[0].to(self.device)
                ohe_batch = loader_data[1].to(self.device).float()  # Batch label
                ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

                optimizer.zero_grad()

                # Encode and decode the input data along with the one-hot encoded batch label
                q_zx = self.vae.encoder(
                    torch.cat([x, ohe_batch], dim=1)
                )  # td.Distribution
                z = q_zx.rsample()  # latent points
                p_xz = self.vae.decoder(
                    torch.cat([z, ohe_batch], dim=1)
                )  # td.Distribution

                # Compute the reconstruction loss (Negative log-likelihood)
                recon_loss = (w_elbo_nll) * -p_xz.log_prob(x).mean()

                # Compute the KL-divergence loss
                kl_loss = (w_elbo_kl) * self.vae.kl_divergence(
                    torch.cat([x, ohe_batch], dim=1), k_ohe=ohe_bio
                ).mean()  # KL divergence function first encodes the input data

                # Compute extra penalization term for clustering priors
                bio_penalty = 0.0  # ensures points from the same biological group to be mapped on the same cluster
                cluster_penalty = 0.0  # ensures gaussian components to not overlap

                if isinstance(self.vae.prior, MoCPPrior):
                    # Compute penalty for biological mapping
                    pred_bio, _, _ = self.vae.encoder.encode(
                        torch.cat([x, ohe_batch], dim=1)
                    )

                    bio_penalty += (w_bio_penalty) * F.cross_entropy(
                        pred_bio, ohe_bio.argmax(dim=1)
                    )

                    # Compute penalty for group clusters
                    cluster_penalty += (
                        w_cluster_penalty
                    ) * self.vae.prior.cluster_loss()

                # Total loss is reconstruction loss + KL divergence loss
                loss = recon_loss + kl_loss + bio_penalty + cluster_penalty

                # Backpropagation and optimization step
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix(
                    vae_loss=f"{total_loss:.4f}",
                    bio_penalty=f"{bio_penalty:.4f}",
                    clustering_loss=f"{cluster_penalty:.4f}",
                    elbo=f"{(recon_loss + kl_loss):.4f}",
                    epoch=f"{epoch}/{self.phase_1_epochs + 1}",
                )
                progress_bar.update()

        progress_bar.close()

    def batch_correct(
        self,
        train_loader,
        vae_optimizer,
        disc_optimizer,
        adv_optimizer,
        w_disc=1.0,
        w_adv=1.0,
        w_elbo_nll=1.0,
        w_elbo_kl=1.0,
        w_bio_penalty=1.0,
        w_cluster_penalty=1.0,
    ):
        """
        Train the conditional VAE model for batch correction. This is trained after VAE prior parameters are defined,
        """
        self.vae.train()

        total_steps = len(train_loader) * self.phase_2_epochs
        progress_bar = tqdm(
            range(total_steps),
            desc="Training: Embeddings batch effect correction using adversrial training",
        )

        for epoch in range(self.phase_2_epochs):
            total_loss = 0.0
            data_iter = iter(train_loader)
            for loader_data in data_iter:
                x = loader_data[0].to(self.device)
                ohe_batch = loader_data[1].to(self.device).float()  # Batch label
                ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

                # 1. Forward pass latent points to discriminator
                disc_optimizer.zero_grad()
                with torch.no_grad():
                    q_zx = self.vae.encoder(
                        torch.cat([x, ohe_batch], dim=1)
                    )  # td.Distribution
                    z = q_zx.rsample()  # latent points

                pred_batch = self.disc(torch.cat([z, ohe_bio], dim=1))
                disc_loss = (w_disc) * self.disc.loss(
                    pred_batch, ohe_batch.argmax(dim=1)
                )

                # 2. Backpropagation and optimization step for discriminator
                disc_loss.backward()
                disc_optimizer.step()

                # 3. Adversarial backpropagation and optimization step for encoder
                adv_optimizer.zero_grad()
                q_zx = self.vae.encoder(
                    torch.cat([x, ohe_batch], dim=1)
                )  # td.Distribution
                z = q_zx.rsample()  # latent points
                pred_batch = self.disc(torch.cat([z, ohe_bio], dim=1))
                disc_loss = self.disc.loss(pred_batch, ohe_batch.argmax(dim=1))
                adv_loss = (w_adv) * -disc_loss
                adv_loss.backward()
                adv_optimizer.step()

                # 4. Forward pass through VAE
                vae_optimizer.zero_grad()
                q_zx = self.vae.encoder(
                    torch.cat([x, ohe_batch], dim=1)
                )  # td.Distribution
                z = q_zx.rsample()  # latent points
                p_xz = self.vae.decoder(
                    torch.cat([z, ohe_batch], dim=1)
                )  # td.Distribution

                # Compute the reconstruction loss (Negative log-likelihood)
                recon_loss = (w_elbo_nll) * -p_xz.log_prob(x).mean()

                # Compute the KL-divergence loss
                kl_loss = (w_elbo_kl) * self.vae.kl_divergence(
                    torch.cat([x, ohe_batch], dim=1), k_ohe=ohe_bio
                ).mean()  # KL divergence function first encodes the input data

                # Compute extra penalization term for clustering priors
                bio_penalty = 0.0  # ensures points from the same biological group to be mapped on the same cluster
                cluster_penalty = 0.0  # ensures gaussian components to not overlap

                if isinstance(self.vae.prior, MoCPPrior):
                    # Compute penalty for biological mapping
                    pred_bio, _, _ = self.vae.encoder.encode(
                        torch.cat([x, ohe_batch], dim=1)
                    )

                    bio_penalty += (w_bio_penalty) * F.cross_entropy(
                        pred_bio, ohe_bio.argmax(dim=1)
                    )

                    # Compute penalty for group clusters
                    cluster_penalty += (
                        w_cluster_penalty
                    ) * self.vae.prior.cluster_loss()

                # Total loss is reconstruction loss + KL divergence loss
                vae_loss = recon_loss + kl_loss + bio_penalty + cluster_penalty

                # Backpropagation and optimization step
                vae_loss.backward()
                vae_optimizer.step()

                total_loss += vae_loss.item()

                # Update progress bar
                progress_bar.set_postfix(
                    vae_loss=f"{total_loss:.4f}",
                    bio_penalty=f"{bio_penalty:.4f}",
                    clustering_loss=f"{cluster_penalty:.4f}",
                    elbo=f"{(recon_loss + kl_loss):.4f}",
                    disc_loss=f"{disc_loss:.4f}",
                    adv_loss=f"{adv_loss:.4f}",
                    epoch=f"{epoch}/{self.phase_2_epochs + 1}",
                )
                progress_bar.update()

        progress_bar.close()

    def batch_mask(
        self,
        train_loader,
        decoder_optimizer,
        smooth_annealing=True,
        cycle_reg=None,
        w_elbo_nll=1.0,
        w_cycle=1e-3,
    ):
        """
        Pre-trained VAE will now have frozen encoder and batch labels masked at the encoder.
        """

        self.vae.train()

        total_steps = len(train_loader) * self.phase_3_epochs
        progress_bar = tqdm(
            range(total_steps), desc="Training: VAE decoder with masked batch labels"
        )

        for epoch in range(self.phase_3_epochs):
            # Introduce slow transition to full batch masking
            if smooth_annealing:
                alpha = max(0.0, 1.0 - (2 * epoch / self.phase_3_epochs))
            else:
                alpha = 0.0

            data_iter = iter(train_loader)
            for loader_data in data_iter:
                x = loader_data[0].to(self.device)
                ohe_batch = loader_data[1].to(self.device).float()  # Batch label
                ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

                # VAE ELBO computation with masked batch label
                decoder_optimizer.zero_grad()

                # Forward pass to encoder
                q_zx = self.vae.encoder(torch.cat([x, ohe_batch], dim=1))

                # Sample from encoded point
                z = q_zx.rsample()

                # Forward pass to the decoder
                p_xz = self.vae.decoder(
                    torch.cat([z, alpha * ohe_batch], dim=1)
                )  # masked batch label

                # Compute the reconstruction loss (Negative log-likelihood)
                recon_loss = (w_elbo_nll) * -p_xz.log_prob(x).mean()

                # Cycle loss for regularization (reconstructed point should be mapped to same cluster)
                if cycle_reg is not None:
                    # Reconstruct point
                    recon_x = p_xz.sample()
                    recon_z_params = self.vae.encoder.encode(
                        [recon_x, ohe_batch], dim=1
                    )

                    recon_pi = (
                        recon_z_params[0]
                        if isinstance(self.vae.prior, MoCPPrior)
                        else None
                    )

                    cycle_loss = (
                        (w_cycle) * F.cross_entropy(recon_pi, ohe_bio.argmax(dim=1))
                        if recon_pi is not None
                        else 0
                    )

                else:
                    cycle_loss = 0

                # Compute loss
                vae_loss = recon_loss + cycle_loss
                vae_loss.backward()
                decoder_optimizer.step()

                # Update progress bar
                progress_bar.set_postfix(
                    vae_loss=f"{vae_loss:12.4f}",
                    cycle_loss=f"{cycle_loss:12.4f}",
                    epoch=f"{epoch + 1}/{self.phase_3_epochs}",
                )
                progress_bar.update()

        progress_bar.close()

    def fit(
        self,
        smooth_annealing=True,
        cycle_reg=None,
        seed=None,
        # VAE Model parameters
        w_elbo_nll=1.0,
        w_elbo_kl=1.0,
        w_bio_penalty=1.0,
        w_cluster_penalty=1.0,
        w_cycle=1e-3,
        # Batch discriminator parameters
        w_disc=1.0,
        w_adv=1.0,
        # Optimizers learning rates
        phase_1_vae_lr=1e-3,
        phase_2_vae_lr=1e-3,
        phase_3_vae_lr=1e-6,
        disc_lr=1e-3,
        adv_lr=1e-3,
    ):
        # Define optimizer
        if isinstance(self.vae.prior, MoCPPrior):
            prior_params = self.vae.prior.parameters()

        elif isinstance(self.vae.prior, VMMPrior):
            prior_params = [self.vae.prior.u, self.vae.prior.var]

        else:
            raise NotImplementedError(
                "metaABaCo prior distribution can only be 'MoG' or 'VMM'"
            )

        vae_optimizer_1 = torch.optim.Adam(
            [
                {"params": self.vae.encoder.parameters()},
                {"params": self.vae.decoder.parameters()},
                {"params": prior_params},
            ],
            lr=phase_1_vae_lr,
        )

        vae_optimizer_2 = torch.optim.Adam(
            [
                {"params": self.vae.encoder.parameters()},
                {
                    "params": self.vae.decoder.parameters()
                },  # only VAE weights are updated, prior distribution is fixed on this stage
            ],
            lr=phase_2_vae_lr,
        )

        vae_optimizer_3 = torch.optim.Adam(
            [
                {
                    "params": self.vae.decoder.parameters()
                },  # onlye Decoder weights are updated, Encoder is fixed on this stage
            ],
            lr=phase_3_vae_lr,
        )

        disc_optimizer = torch.optim.Adam(
            self.disc.parameters(),
            lr=disc_lr,
        )

        adv_optimizer = torch.optim.Adam(
            [
                {"params": self.vae.encoder.parameters()},
            ],
            lr=adv_lr,
        )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # PHASE 1: Train VAE for reconstructing data and getting meaningful embeddings
        self.train_vae(
            self.dataloader,
            vae_optimizer_1,
            w_elbo_nll,
            w_elbo_kl,
            w_bio_penalty,
            w_cluster_penalty,
        )
        # PHASE 2: Batch effect correction on the latent space with learned prior distribution
        self.batch_correct(
            self.dataloader,
            vae_optimizer_2,
            disc_optimizer,
            adv_optimizer,
            w_disc,
            w_adv,
            w_elbo_nll,
            w_elbo_kl,
            w_bio_penalty,
            w_cluster_penalty,
        )
        # PHASE 3: Batch masking to the decoder to reconstruct data without batch effect
        self.batch_mask(
            self.dataloader,
            vae_optimizer_3,
            smooth_annealing,
            cycle_reg,
            w_elbo_nll,
            w_cycle,
        )

    def correct(
        self,
        seed=None,
        mask=True,
    ):
        self.vae.eval()

        recon_data = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        for loader_data in iter(self.dataloader):
            x = loader_data[0].to(self.device)
            ohe_batch = loader_data[1].to(self.device).float()  # Batch label
            # ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

            # Encode and decode the input data along with the one-hot encoded batch label
            q_zx = self.vae.encoder(torch.cat([x, ohe_batch], dim=1))  # td.Distribution
            z = q_zx.rsample()  # latent points
            if mask:
                p_xz = self.vae.decoder(
                    torch.cat([z, torch.zeros_like(ohe_batch.to(self.device))], dim=1)
                )  # td.Distribution
            else:
                p_xz = self.vae.decoder(
                    torch.cat([z, ohe_batch], dim=1)
                )  # useful when there is no batch effect to correct

            # Sample from the output distribution
            x_recon = p_xz.sample()  # Reconstructed data

            # Rebuild the input data format for analysis
            recon_data.append(x_recon.cpu().detach().numpy())

        np_recon_data = np.vstack([t for t in recon_data])

        x_recon_data = pd.concat(
            [
                self.data.select_dtypes(exclude="number"),
                pd.DataFrame(
                    np_recon_data,
                    index=self.data.index,
                    columns=self.data.select_dtypes("number").columns,
                ),
            ],
            axis=1,
        )

        return x_recon_data

    def plot_pca_posterior(self, figsize=(14, 6), palette="tab10"):
        """
        Get the plot of the first 2 principal components of the posterior distribution.
        """
        self.vae.eval()
        z_pca = []
        for loader_data in iter(self.dataloader):
            x = loader_data[0].to(self.device)
            ohe_batch = loader_data[1].to(self.device).float()  # Batch label
            ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

            l1 = ohe_batch.detach().cpu().numpy().argmax(axis=1)
            l2 = ohe_bio.detach().cpu().numpy().argmax(axis=1)

            coords = self.vae.pca_posterior(torch.cat([x, ohe_batch], dim=1))
            z_pca.append(coords)

        coords = np.vstack(coords)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=l1,
            palette=palette,
            ax=axes[0],
            legend="full",
        )
        axes[0].set_title("Posterior PCA colored by batch group")
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=l2,
            palette=palette,
            ax=axes[1],
            legend="full",
        )
        axes[1].set_title("Posterior PCA colored by biological group")
        plt.tight_layout()
        plt.show()
