import math
import numpy as np
import torch
from torch import nn
from scheduler import ConstantScheduler, ConstrainedExponentialSchedulerMaLagrange
from simple_net.My_latent import MyFixedGaussian, MyGaussian, FusionModel, MyMLPEncoder, MyMLPDecoder
from simple_net.initializer import initialize_weight
from utils import calculate_kl_divergence


class MyLatentModel(nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics.
    """

    def __init__(
            self,
            state_shape=(12,),
            info_shape=(4,),
            action_shape=(1,),
            feature_dim=256,
            z1_dim=32,
            z2_dim=256,
            hidden_units=(256, 256),
            scheduler_recon=ConstrainedExponentialSchedulerMaLagrange,
            scheduler_classifier=ConstrainedExponentialSchedulerMaLagrange,
            scheduler_kl=ConstantScheduler,
    ):
        super(MyLatentModel, self).__init__()
        # p(z1(0)) = N(0, I)
        self.z1_prior_init = MyFixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = MyGaussian(z1_dim, z2_dim, hidden_units)
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = MyGaussian(
            z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = MyGaussian(
            z1_dim + z2_dim + action_shape[0],
            z2_dim,
            hidden_units,
        )

        # q(z1(0) | feat(0))
        self.z1_posterior_init = MyGaussian(feature_dim, z1_dim, hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = MyGaussian(
            feature_dim + z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        self.first_encoder = FusionModel(info_shape[0], state_shape[0], feature_dim)
        # feat(t) = MyEncoder(x(t))
        self.encoder = MyMLPEncoder(state_shape[0], feature_dim)
        # p(x(t) | z1(t), z2(t))
        self.decoder = MyMLPDecoder(
            z1_dim + z2_dim,
            state_shape[0],
            std=np.sqrt(0.1),
        )
        self.apply(initialize_weight)

        self.scheduler_recon = scheduler_recon
        self.scheduler_classifier = scheduler_classifier
        self.scheduler_kl = scheduler_kl

    def sample_prior(self, actions_, z2_post_):
        # p(z1(0)) = N(0, I)
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # p(z1(t) | z2(t-1), a(t-1))
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        # Concatenate initial and consecutive latent variables
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return z1_mean_, z1_std_

    def sample_posterior(self, features_, actions_):
        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, actions_.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return z1_mean_, z1_std_, z1_, z2_

    def calculate_loss(self, state_, action_, pred, label):
        # Calculate the sequence of features.
        feature_ = self.encoder(state_)

        # Sample from latent variable model.
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # Prediction loss of signals.
        z_ = torch.cat([z1_, z2_], dim=-1)
        state_mean_, state_std_ = self.decoder(z_)
        # classification + regression
        state_noise_ = (state_ - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_recon = -log_likelihood_.mean(dim=0).sum()

        loss_supervised = torch.mean(torch.pow((pred - label), 2)) / torch.mean(torch.pow(label, 2))

        return loss_kld, loss_recon, loss_supervised

    def forward(self, state_, info_, action_):
        # Calculate the sequence of features.
        # state.shape = (B, 6, 12)
        initial_state = torch.permute(state_[:, 0:1, :], [0, 2, 1])
        initial_feature_ = self.first_encoder(info_[:, :, 0], initial_state[:, :, 0])
        # (12, 1, 256)
        # (12, 5, 256)
        feature_ = self.encoder(state_[:, 1:, :])
        all_feature = torch.cat([torch.unsqueeze(initial_feature_, dim=1), feature_], dim=1)
        # Sample from latent variable model.
        # z1_.shape = (12, 6, 32) z2_.shape = (12, 6, 256)
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(all_feature, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # z_.shape = (12, 6, z1+z2)
        z_ = torch.cat([z1_, z2_], dim=-1)
        # pred_mean.shape = pred_sta.shape = (12, 6, 12)
        pred_mean_, pred_std_ = self.decoder(z_)

        return z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_, pred_mean_, pred_std_

    def getloss(self, z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_, pred_mean, pred_std, label):
        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # Prediction loss of images.
        # classification + regression
        pred_noise_ = (label - pred_mean[:, 1:, :]) / (pred_std[:, 1:, :] + 1e-8)
        log_likelihood_ = (-0.5 * pred_noise_.pow(2) - pred_std[:, 1:, :].log()) - 0.5 * math.log(2 * math.pi)
        loss_recon = -log_likelihood_.mean(dim=0).sum()

        # pred = pred_mean + torch.randn_like(pred_std) * pred_std
        # loss_supervised = torch.mean(torch.pow((pred - label), 2)) / torch.mean(torch.pow(label, 2))

        # only update in the training step
        if self.training or (self.lambda_recon is None):
            self.lambda_recon = self.scheduler_recon(float(loss_recon.detach()))

        if self.training or (self.lambda_kl is None):
            self.lambda_kl = self.scheduler_kl(float(loss_kld.detach()))

        if self.training or (self.lambda_classifier is None):
            self.lambda_classifier = self.scheduler_classifier(
                float(loss_classification.detach())
            )

        loss = (
                self.lambda_classifier * loss_recon
                + self.lambda_kl * loss_kld
                + self.lambda_classifier * loss_classification
        )

        return loss_kld, loss_recon
