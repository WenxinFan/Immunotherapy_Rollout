import math
import numpy as np
import torch
from torch import nn

from utils import calculate_kl_divergence


def save_prediction(model, data_loader):
    # toggle model to eval mode
    model.eval()

    kld_list = []
    recon_list = []
    classify_list = []

    # turn off gradients since they will not be used here
    with torch.no_grad():
        for x, act, info, y_tg in data_loader:
            x = x.cuda()
            act = act.cuda()
            info = info.cuda()
            y_tg = y_tg.cuda()
            y = model(x, info, act)
            # pred = pred_mean + torch.randn_like(pred_std) * pred_std
            loss_kld, loss_recon, loss_classification = getLoss(y[0], y[1], y[2], y[3], y[4], y[5], y_tg)
            kld_list.append(loss_kld.cpu().detach().numpy())
            recon_list.append(loss_recon.cpu().detach().numpy())
            classify_list.append(loss_classification.cpu().detach().numpy())

    return np.mean(kld_list), np.mean(recon_list), np.mean(classify_list)


# def rollout(model, data_loader):
#     with torch.no_grad():
#         for x, act, info, y_all in data_loader:
#             x = x.cuda()
#             act = act.cuda()
#             info = info.cuda()
#             y_all = y_all.cuda()
#             for i in range(x.shape[1]-1):
#                 z1_i =


def getLoss(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_, pred_mean, pred_std, label):
    # Calculate KL divergence loss.
    loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

    # Prediction loss of images.
    pred_noise_ = (label - pred_mean[:, 1:, :]) / (pred_std[:, 1:, :] + 1e-8)
    log_likelihood_ = (-0.5 * pred_noise_.pow(2) - pred_std[:, 1:, :].log()) - 0.5 * math.log(2 * math.pi)
    loss_recon = -log_likelihood_.mean(dim=0).sum()

    pred_act = pred_mean[:, 1:, 0:1] + torch.randn_like(pred_std[:, 1:, 0:1]) * pred_std[:, 1:, 0:1]
    binary_loss = nn.BCEWithLogitsLoss()
    loss_classification = binary_loss(pred_act[:, -1, 0:1], label[:, -1, 0:1])

    return loss_kld, loss_recon, loss_classification
