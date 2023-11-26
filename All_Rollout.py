import argparse
import math
import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, r2_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Our_Dataset import Our_Dataset
from simple_net.My_latent import MyFixedGaussian, FusionModel, MyMLPEncoder, MyGaussian, MyMLPDecoder
from simple_net.initializer import initialize_weight
from utils import build_mlp, calculate_kl_divergence

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

parser = argparse.ArgumentParser(description="Training Latent Model")
parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")

parser.add_argument("--feature_dim", type=float, default=256)
parser.add_argument("--z1_dim", type=float, default=32)
parser.add_argument("--z2_dim", type=float, default=256)
parser.add_argument("--state_shape", type=tuple, default=(12,))
parser.add_argument("--info_shape", type=tuple, default=(4,))
parser.add_argument("--action_shape", type=tuple, default=(1,))
parser.add_argument("--num_sequence", type=int, default=5)

parser.add_argument("--data_path", type=str, default='/home/fan/Immunotherapy/MyData/')
parser.add_argument("--result_path", type=str, default='/home/fan/Immunotherapy/Results/')
parser.add_argument("--model_path", type=str, default='/home/fan/Immunotherapy/Models/')
opt = parser.parse_args()


class AllRolloutModel(nn.Module):
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
    ):
        super(AllRolloutModel, self).__init__()
        self.sequence_length = sequence_length
        self.all_actions = True
        self.pred_actions = []
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
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        return z1_mean_, z1_std_

    def sample_posterior(self, features_, actions_):
        # z1_.shape=(2, 32)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # z2_.shape=(2, 256)
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

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
        state.shape = (B, 5, 12) 此时只利用初始的状态信息
        info.shape = (B, 4, 1)
        action.shape = (B, 5, 1)
        """
        initial_state = torch.permute(state_[:, 0:1, :], [0, 2, 1])
        # (B, 256)
        initial_feature_ = self.first_encoder(info_[:, :, 0], initial_state[:, :, 0])
        # (B, 32) (B, 256)
        _, _, z1_, z2_ = self.sample_posterior(torch.unsqueeze(initial_feature_, dim=1), action_[:, 0:1, :])

        z_ = torch.cat([z1_, z2_], dim=-1)
        # pred_mean.shape = pred_sta.shape = (B, 1, 13)
        pred_mean_, pred_std_ = self.decoder(z_)
        pred = pred_mean_ + torch.randn_like(pred_std_) * pred_std_

        single_batch_act = []

        for i in range(1, self.sequence_length):
            current_state = pred[:, :, 1:]
            if self.all_actions:
                current_act = action_[:, i:i + 1, :]
            else:
                current_act = pred[:, :, 0:1]
                single_batch_act.append(current_act)
            # (B, 1, 256)
            feature_ = self.encoder(current_state)
            z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, current_act)
            z_ = torch.cat([z1_, z2_], dim=-1)
            pred_mean_, pred_std_ = self.decoder(z_)
            pred = pred_mean_ + torch.randn_like(pred_std_) * pred_std_

        last_state = pred[:, :, 1:]
        last_act = pred[:, :, 0:1]
        if not self.all_actions:
            single_batch_act = np.concatenate(single_batch_act, axis=1)
            self.pred_actions.append(single_batch_act)

        return last_state, last_act


def main():
    # Load dataset
    print('Loading Dataset ...\n')
    data_path = opt.data_path
    dataset_test = Our_Dataset(data_path, data_type='test')

    # training data loader
    device_id = [0, 1, 2, 3]
    b_size = len(device_id) * opt.batch_size

    # validation data loader
    val_loader = DataLoader(dataset=dataset_test, num_workers=32, batch_size=b_size, shuffle=False)
    print('{} validation samples'.format(len(dataset_test)))
    print('{} validation batches'.format(len(val_loader)))

    val_model_path = '/home/fan/Immunotherapy/Models/epoch1_lr0.0001_slac.pth'
    val_net = AllRolloutModel(opt.state_shape, opt.info_shape, opt.action_shape, opt.feature_dim, opt.z1_dim,
                              opt.z2_dim, hidden_units=(256, 256))

    # if all_actions=True: 给定治疗前的状态评分和过程中的所有action 预测治疗后的状态评分
    # else: 给定治疗前的状态评分，初始action (初始action总为1) 以及基础信息
    # 判断患者（1）患者能否坚持到最后 （2）如果中途放弃预测患者会在什么时间点中断 (是否需要修改loss让模型学习中断时间点) (3)也需要预测只给定初始action的状态评分
    # (可以对比在不同时间点中断后 最后的状态评分 得出一个对患者而言最佳的治疗时间)

    val_net.all_actions = False

    val_model = nn.DataParallel(val_net.cuda(), device_ids=device_id).cuda()
    val_model.load_state_dict(torch.load(val_model_path))

    pred_states = []
    pred_acts = []

    label_states = []
    label_acts = []

    with torch.no_grad():
        val_model.eval()
        for x, act, info, y_all in val_loader:
            x = x.cuda()
            act = act.cuda()
            info = info.cuda()
            # y_all = y_all.cuda()
            pred_state, pred_act = val_model(x, info, act)
            pred_states.append(torch.squeeze(pred_state).cpu().numpy())
            pred_acts.append(torch.squeeze(pred_act).cpu().numpy())

            label_states.append(torch.squeeze(y_all[:, -1, 1:]).cpu().numpy())
            label_acts.append(torch.squeeze(y_all[:, -1, 0:1]).cpu().numpy())

    # classification accuracy
    pred_acts = np.array(pred_acts, dtype=object)
    pred_acts = np.concatenate((pred_acts[:, 0], pred_acts[:, 1]), axis=0)
    pred_acts = np.where(pred_acts > 0.5, 1, 0)
    pred_acts = np.array(pred_acts, dtype=int)
    label_acts = np.array(label_acts, dtype=object)
    label_acts = np.concatenate((label_acts[:, 0], label_acts[:, 1]), axis=0)
    label_acts = np.array(label_acts, dtype=int)

    fpr, tpr, thresholds = roc_curve(label_acts, pred_acts)
    auc_score = roc_auc_score(label_acts, pred_acts)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')  # 设置标题
    plt.legend()
    plt.show()

    # regression accuracy
    pred_states = np.array(pred_states, dtype=object)
    pred_states = np.concatenate((pred_states[:, 0, :], pred_states[:, 1, :]), axis=0)
    pred_states = np.nan_to_num(pred_states)
    label_states = np.array(label_states, dtype=object)
    label_states = np.concatenate((label_states[:, 0], label_states[:, 1]), axis=0)
    mse = mean_squared_error(label_states, pred_states)
    rmse = np.sqrt(mean_squared_error(label_states, pred_states))
    r2 = r2_score(label_states, pred_states)
    print('mse={:.5f}\t rmse={:.5f}\t r2={:.5f}\t'.format(mse, rmse, r2))

    # 中断时间点（测试中断时间的精度）
    predicted_acts = np.array(val_net.pred_actions, dtype=object)
    # predicted_acts = np.concatenate((predicted_acts[0], predicted_acts[1]), axis=0)
    break_timestamp = []
    for i in range(predicted_acts.shape[0]):
        single_act = predicted_acts[i]
        for j in range(single_act.shape[1]):
            #
            if single_act[:, j, :] < 0.5:
                break_timestamp.append(j + 2)
            if j == 3:
                break_timestamp.append(5)


def val_Rollout(device_id, val_model_path, val_loader):
    val_net = AllRolloutModel(opt.state_shape, opt.info_shape, opt.action_shape, opt.feature_dim, opt.z1_dim,
                              opt.z2_dim, hidden_units=(256, 256))

    val_net.all_actions = True

    val_model = nn.DataParallel(val_net.cuda(), device_ids=device_id).cuda()
    val_model.load_state_dict(torch.load(val_model_path))

    pred_states = []
    pred_acts = []

    label_states = []
    label_acts = []

    with torch.no_grad():
        val_model.eval()
        for x, act, info, y_all in val_loader:
            x = x.cuda()
            act = act.cuda()
            info = info.cuda()
            # y_all = y_all.cuda()
            pred_state, pred_act = val_model(x, info, act)
            pred_states.append(torch.squeeze(pred_state).cpu().numpy())
            pred_acts.append(torch.squeeze(pred_act).cpu().numpy())

            label_states.append(torch.squeeze(y_all[:, -1, 1:]).cpu().numpy())
            label_acts.append(torch.squeeze(y_all[:, -1, 0:1]).cpu().numpy())

    # classification accuracy
    pred_acts = np.array(pred_acts, dtype=object)
    pred_acts = np.concatenate((pred_acts[0], pred_acts[1], pred_acts[2], pred_acts[3]), axis=0)
    pred_acts = np.where(pred_acts > 0.5, 1, 0)
    pred_acts = np.array(pred_acts, dtype=int)
    label_acts = np.array(label_acts, dtype=object)
    label_acts = np.concatenate((label_acts[0], label_acts[1], label_acts[2], label_acts[3]), axis=0)
    label_acts = np.array(label_acts, dtype=int)

    auc_score = roc_auc_score(label_acts, pred_acts)

    # regression accuracy
    pred_states = np.array(pred_states, dtype=object)
    pred_states = np.concatenate((pred_states[0], pred_states[1], pred_states[2], pred_states[3]), axis=0)
    pred_states = np.nan_to_num(pred_states)
    label_states = np.array(label_states, dtype=object)
    label_states = np.concatenate((label_states[0], label_states[1], label_states[2], label_states[3]), axis=0)
    mse = mean_squared_error(label_states, pred_states)
    rmse = np.sqrt(mean_squared_error(label_states, pred_states))
    r2 = r2_score(label_states, pred_states)
    return auc_score, mse, rmse, r2


if __name__ == "__main__":
    main()
