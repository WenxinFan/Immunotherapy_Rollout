import math
import numpy as np
import torch
import torch.nn as nn
from simple_net.My_latent import MyFixedGaussian, FusionModel, MyMLPEncoder, MyGaussian, MyMLPDecoder
from simple_net.initializer import initialize_weight
from simple_net.scheduler import ConstrainedExponentialSchedulerMaLagrange, ConstantScheduler
from utils import build_mlp, calculate_kl_divergence


class LatentModel(nn.Module):
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
            sequence_length=5,
            # annealing_rate need to be tuned
            # constraint_bound是loss的平稳值 kl_lambda先设为常数 如果lambda上下振荡 那么annealing需要调小一点
            # reconstruction的学习率可以设小一点，constraint_bound也相对设小一点，push网络的学习
            # classify的constraint_bound可能需要调高一些
            scheduler_recon=ConstrainedExponentialSchedulerMaLagrange(constraint_bound=500, annealing_rate=1e-4),
            scheduler_kl=ConstantScheduler(lam=1),
            # scheduler_kl=ConstrainedExponentialSchedulerMaLagrange(constraint_bound=500, annealing_rate=1e-4,
            #                                                        lower_bound_lam=0),
            scheduler_classifier=ConstrainedExponentialSchedulerMaLagrange(constraint_bound=1, annealing_rate=1e-4),
    ):
        super(LatentModel, self).__init__()

        self.lambda_classifier = None
        self.lambda_kl = None
        self.lambda_recon = None
        self.sequence_length = sequence_length
        self.scheduler_recon = scheduler_recon
        self.scheduler_classifier = scheduler_classifier
        self.scheduler_kl = scheduler_kl

        self.lambda_recon_list = []
        self.lambda_kl_list = []
        self.lambda_classify_list = []

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
            state_shape[0] + 1,
            std=np.sqrt(0.1),
        )
        self.apply(initialize_weight)

    def sample_prior(self, actions_, z2_post_):
        # z1_init.shape=(2, 32)
        # z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # p(z1(t) | z2(t-1), a(t-1))
        # z2_post_ = torch.unsqueeze(z2_post_, dim=1)
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        # Concatenate initial and consecutive latent variables
        # z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        # z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return z1_mean_, z1_std_

    def sample_posterior(self, features_, actions_):
        # z1_.shape=(2, 32)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # z2_.shape=(2, 256)
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        # z1_mean_ = [z1_mean]
        # z1_std_ = [z1_std]
        # z1_ = [z1]
        # z2_ = [z2]
        # for t in range(actions_.size(1)):
        #     z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t]], dim=1))
        #     z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        #     z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t]], dim=1))
        #     z2 = z2_mean + torch.randn_like(z2_std) * z2_std
        #
        #     z1_mean_.append(z1_mean)
        #     z1_std_.append(z1_std)
        #     z1_.append(z1)
        #     z2_.append(z2)
        # z1_mean_ = torch.stack(z1_mean_, dim=1)
        # z1_std_ = torch.stack(z1_std_, dim=1)
        # z1_ = torch.stack(z1_, dim=1)
        # z2_ = torch.stack(z2_, dim=1)

        z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, 0], z2, actions_[:, 0]], dim=1))
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, 0]], dim=1))
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = torch.unsqueeze(z1_mean, dim=1)
        z1_std_ = torch.unsqueeze(z1_std, dim=1)
        z1_ = torch.unsqueeze(z1, dim=1)
        z2_ = torch.unsqueeze(z2, dim=1)

        return z1_mean_, z1_std_, z1_, z2_

    def forward(self, state_, info_, action_):
        """
        state.shape = (B, 5, 12)
        info.shape = (B, 4, 1)
        action.shape = (B, 5, 1)
        """

        initial_state = torch.permute(state_[:, 0:1, :], [0, 2, 1])
        # (B, 256)
        initial_feature_ = self.first_encoder(info_[:, :, 0], initial_state[:, :, 0])
        # (B, 32) (B, 256)
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(torch.unsqueeze(initial_feature_, dim=1),
                                                                      action_[:, 0:1, :])
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_[:, 0:1, :], z2_)
        z_ = torch.cat([z1_, z2_], dim=-1)
        # pred_mean.shape = pred_sta.shape = (B, 1, 13)
        pred_mean_, pred_std_ = self.decoder(z_)

        z1_mean_post = [z1_mean_post_]
        z1_std_post = [z1_std_post_]
        z1_mean_pri = [z1_mean_pri_]
        z1_std_pri = [z1_std_pri_]
        # z1 = [z1_]
        # z2 = [z2_]
        pred_mean = [pred_mean_]
        pred_std = [pred_std_]

        for i in range(1, self.sequence_length):
            # (B, 1, 256)
            feature_ = self.encoder(state_[:, i:i + 1, :])
            z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_[:, i:i + 1, :])
            z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_[:, i:i + 1, :], z2_)
            z_ = torch.cat([z1_, z2_], dim=-1)
            pred_mean_, pred_std_ = self.decoder(z_)

            z1_mean_post.append(z1_mean_post_)
            z1_std_post.append(z1_std_post_)
            z1_mean_pri.append(z1_mean_pri_)
            z1_std_pri.append(z1_std_pri_)

            # z1.append(z1_)
            # z2.append(z2_)

            pred_mean.append(pred_mean_)
            pred_std.append(pred_std_)

        z1_mean_post = torch.concatenate(z1_mean_post, dim=1)
        z1_std_post = torch.concatenate(z1_std_post, dim=1)
        z1_mean_pri = torch.concatenate(z1_mean_pri, dim=1)
        z1_std_pri = torch.concatenate(z1_std_pri, dim=1)
        # z1 = torch.concatenate(z1, dim=1)
        # z2 = torch.concatenate(z2, dim=1)
        pred_mean = torch.concatenate(pred_mean, dim=1)
        pred_std = torch.concatenate(pred_std, dim=1)

        return z1_mean_post, z1_std_post, z1_mean_pri, z1_std_pri, pred_mean, pred_std

    def getloss(self, z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_, pred_mean, pred_std, label):
        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # Prediction loss of regression.
        pred_noise_ = (label[:, :, 1:] - pred_mean[:, :, 1:]) / (pred_std[:, :, 1:] + 1e-8)
        log_likelihood_ = (-0.5 * pred_noise_.pow(2) - pred_std[:, :, 1:].log()) - 0.5 * math.log(2 * math.pi)
        loss_recon = -log_likelihood_.mean(dim=0).sum()

        # classification loss 每一步的action都需要算loss
        pred_act = pred_mean[:, :, 0:1] + torch.randn_like(pred_std[:, :, 0:1]) * pred_std[:, :, 0:1]
        binary_loss = nn.BCEWithLogitsLoss()
        loss_classification = binary_loss(pred_act, label[:, :, 0:1])

        # only update in the training step
        if self.training or (self.lambda_recon is None):
            self.lambda_recon = self.scheduler_recon(float(loss_recon.detach()))
            self.lambda_recon_list.append(self.lambda_recon)

        if self.training or (self.lambda_kl is None):
            self.lambda_kl = self.scheduler_kl(float(loss_kld.detach()))
            self.lambda_kl_list.append(self.lambda_kl)

        if self.training or (self.lambda_classifier is None):
            self.lambda_classifier = self.scheduler_classifier(float(loss_classification.detach()))
            self.lambda_classify_list.append(self.lambda_classifier)

        loss = (
                self.lambda_recon * loss_recon
                + self.lambda_kl * loss_kld
                + self.lambda_classifier * loss_classification
        )

        # loss = (
        #         0.01 * loss_recon
        #         + 0.1 * loss_kld
        #         + 1 * loss_classification
        # )

        return loss_kld, loss_recon, loss_classification, loss
