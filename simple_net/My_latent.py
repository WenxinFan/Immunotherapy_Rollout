import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from simple_net.initializer import initialize_weight
from utils import build_mlp, calculate_kl_divergence


class MyFixedGaussian(nn.Module):
    """
    Fixed diagonal gaussian distribution.
    """

    def __init__(self, output_dim, std):
        super(MyFixedGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std


class MyGaussian(nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(MyGaussian, self).__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(0.2),
        ).apply(initialize_weight)

    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        return mean, std


class FusionModel(nn.Module):
    def __init__(self, info_dim, state_dim, output_dim, hidden_dim=128):
        super(FusionModel, self).__init__()

        # Define layers for individual information and current state fusion
        self.info_layer = nn.Linear(info_dim, hidden_dim).apply(initialize_weight)
        self.fusion_layer = nn.Linear(hidden_dim + state_dim, output_dim).apply(initialize_weight)
        # self.output_fc = nn.Linear(hidden_dim, output_dim).apply(initialize_weight)
        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, info, initial_state):
        info_out = self.leakyReLU(self.info_layer(info))

        # Concatenate individual information with initial state
        fused_input = torch.cat((info_out, initial_state), dim=1)

        fused_out = self.leakyReLU(self.fusion_layer(fused_input))

        # Output prediction
        # prediction = self.output_fc(fused_out)

        return fused_out


class MyMLPEncoder(nn.Module):
    """
    MyEncoder.
    """

    def __init__(self, input_dim=3, output_dim=5):
        super(MyMLPEncoder, self).__init__()

        # MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)

    # x.shape (12, 5, 12) batch_size=12, time_series_length=5, measures_length=12
    def forward(self, x):
        B, S, L = x.size()
        x = x.contiguous().view(B * S, L)
        x = x.contiguous().view(B * S, L)
        x = self.net(x)
        x = x.view(B, S, -1)
        return x


class MyMLPDecoder(nn.Module):
    """
    MyDecoder.
    """

    def __init__(self, input_dim=288, output_dim=3, std=1.0):
        super(MyMLPDecoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.std = std

    # x.shape (12, 6, 12) B=12, time_series_length=6, latent_dim=32+256
    def forward(self, x):
        B, S, latent_dim = x.size()
        x = x.view(B * S, latent_dim)
        x = self.net(x)
        _, C = x.size()
        x = x.view(B, S, C)
        return x, torch.ones_like(x).mul_(self.std)


class PersonalInfo:
    """
        Data Structure for predict the first z=[z1, z2]
    """

    def __init__(self, age, gender, distance, income):
        super(PersonalInfo, self).__init__()
        self.age = age
        # 'Female': 1, 'Male': 0
        self.gender = gender
        # '>10km': 1, '<10km': 0
        self.distance = distance
        # '<30%': 0, '50%-70%': 1, '>70%': 2
        self.income = income
        self.personal_states = [self.age, self.gender, self.distance]


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
            state_shape[0]+1,
            std=np.sqrt(0.1),
        )
        self.apply(initialize_weight)

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
        # (B, 256)
        # (B, length-1, 256)
        feature_ = self.encoder(state_[:, 1:, :])
        all_feature = torch.cat([torch.unsqueeze(initial_feature_, dim=1), feature_], dim=1)
        # Sample from latent variable model.
        # z1_.shape = (12, 6, 32) z2_.shape = (12, 6, 256)
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(all_feature, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # z_.shape = (12, 6, z1+z2)
        z_ = torch.cat([z1_, z2_], dim=-1)
        # pred_mean.shape = pred_sta.shape = (12, 6, 13)
        pred_mean_, pred_std_ = self.decoder(z_)

        return z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_, pred_mean_, pred_std_

    def getloss(self, z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_, pred_mean, pred_std, label):
        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # Prediction loss of regression.
        pred_noise_ = (label[:, :, 1:] - pred_mean[:, 1:, 1:]) / (pred_std[:, 1:, 1:] + 1e-8)
        log_likelihood_ = (-0.5 * pred_noise_.pow(2) - pred_std[:, 1:, 1:].log()) - 0.5 * math.log(2 * math.pi)
        loss_recon = -log_likelihood_.mean(dim=0).sum()

        # classification loss
        pred_act = pred_mean[:, 1:, 0:1] + torch.randn_like(pred_std[:, 1:, 0:1]) * pred_std[:, 1:, 0:1]
        binary_loss = nn.BCEWithLogitsLoss()
        loss_classification = binary_loss(pred_act[:, -1, 0:1], label[:, -1, 0:1])

        return loss_kld, loss_recon, loss_classification


class MyEncoder(nn.Module):
    """
    MyEncoder.
    """

    def __init__(self, input_dim=3, output_dim=5):
        super(MyEncoder, self).__init__()

        # MLP
        self.net = nn.Sequential(
            # (13, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)

    def forward(self, x):
        B, S, L = x.size()
        x = x.view(B * S, L, 1, 1)
        x = self.net(x)
        x = x.view(B, S, -1)
        return x


class MyDecoder(nn.Module):
    """
    MyDecoder.
    """

    def __init__(self, input_dim=288, output_dim=3, std=1.0):
        super(MyDecoder, self).__init__()

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (3, 1, 1)
            nn.ConvTranspose2d(32, output_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ).apply(initialize_weight)
        self.std = std

    def forward(self, x):
        B, S, latent_dim = x.size()
        x = x.view(B * S, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(B, S, C, W, H)
        return x, torch.ones_like(x).mul_(self.std)
