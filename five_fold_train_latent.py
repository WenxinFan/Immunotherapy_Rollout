import argparse
import os
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from All_Rollout import AllRolloutModel, val_Rollout
from Our_Dataset import Our_Dataset
from common import save_prediction
# from simple_net.My_latent import MyLatentModel
from simple_net.Latent import LatentModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5"

parser = argparse.ArgumentParser(description="Five Fold Training Latent Model")
parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")

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


def main():
    device_id = [0, 1, 2, 3]
    b_size = len(device_id) * opt.batch_size
    # Load dataset
    print('Loading Dataset ...\n')
    data_path = opt.data_path
    dataset_train = Our_Dataset(data_path, data_type='train')
    dataset_test = Our_Dataset(data_path, data_type='test')

    train_kfold = torch.utils.data.random_split(dataset_train, [0.2, 0.2, 0.2, 0.2, 0.2],
                                                generator=torch.Generator().manual_seed(0))

    # test data loader
    test_loader = DataLoader(dataset=dataset_test, num_workers=32, batch_size=b_size, shuffle=False)
    print('{} test samples'.format(len(dataset_test)))
    print('{} test batches'.format(len(test_loader)))

    pre_prefix = '_epoch' + str(opt.epochs) + '_lr' + str(opt.lr)

    # training
    running_loss = 0
    fold_loss = []

    for fold in range(5):
        train_data = torch.utils.data.ConcatDataset([train_kfold[i] for i in range(5) if i != fold])
        val_data = train_kfold[fold]
        train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=b_size, shuffle=False)

        train_loss = []
        recon_list = []
        kl_list = []
        classify_list = []
        min_loss = math.inf

        prefix = 'fold_' + str(fold) + pre_prefix
        net = LatentModel(opt.state_shape, opt.info_shape, opt.action_shape, opt.feature_dim, opt.z1_dim, opt.z2_dim,
                          hidden_units=(256, 256))
        criterion = net.getloss
        model = nn.DataParallel(net.cuda(), device_ids=device_id).cuda()
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

        for epoch in tqdm(range(opt.epochs)):
            model.train()
            loss_list = []
            loss_kld_list = []
            loss_recon_list = []
            loss_classify_list = []

            for x_states, acts, info, y_all in train_loader:
                # training step
                model.zero_grad()
                optimizer.zero_grad()

                # [B, 5, 12]
                x_states = x_states.cuda()

                # [B, 5, 12]
                # y_states = y_states.cuda()

                # [B, 5, 13] acts+y_states
                y_all = y_all.cuda()

                # [B, 5, 1]
                acts = acts.cuda()

                # [B, 4, 1]
                info = info.cuda()

                # {z1_post[B, 5, 32], [B, 5, 32], z1_pri[B, 5, 32], [B, 5, 32], pred_mean[B, 5, 12], pred_std[B, 5, 12]}
                y = model(x_states, info, acts)

                loss_kld, loss_recon, loss_classification, loss = criterion(y[0], y[1], y[2], y[3], y[4], y[5], y_all)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                loss_kld_list.append(loss_kld.item())
                loss_recon_list.append(loss_recon.item())
                loss_classify_list.append(loss_classification.item())

            # at the end of each epoch
            loss = np.mean(loss_list)
            running_loss += loss
            train_loss.append(loss)
            recon_list.append(np.mean(loss_recon_list))
            kl_list.append(np.mean(loss_kld_list))
            classify_list.append(np.mean(loss_classify_list))
            if epoch % 100 == 0:
                print('Epoch ' + str(epoch + 1) + ' / ' + str(opt.epochs) + ' loss: ' + str(loss))
            # print('Loss: ' + str(loss))
            # print('Loss KL Distance: ' + str(np.mean(loss_kld_list)))
            # print('Loss Regression: ' + str(np.mean(loss_recon_list)))
            # print('Loss Classification: ' + str(np.mean(loss_classify_list)))

            if loss < min_loss:
                # save model
                model_path = opt.model_path
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(model.state_dict(), opt.model_path + '/' + prefix + '_slac.pth')

        # at the end of each fold 应该用val_data验证一下性能，
        # 是取平均值还是取最大值 box plot（确认一下five-fold的使用）
        fold_loss.append(train_loss[-1])
        print('Fold ' + str(fold + 1) + ' loss: ' + str(train_loss[-1]))
        val_model_path = opt.model_path + '/' + prefix + '_slac.pth'
        auc_score, mse, rmse, r2 = val_Rollout(device_id, val_model_path, val_loader)
        print('AUC={:.5f}\t MSE={:.5f}\t RMSE={:.5f}\t R2={:.5f}\t'.format(auc_score, mse, rmse, r2))


if __name__ == "__main__":
    main()
