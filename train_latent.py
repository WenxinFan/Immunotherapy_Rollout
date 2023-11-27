import argparse
import os
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from All_Rollout import val_Rollout, AllRolloutModel
from Our_Dataset import Our_Dataset
from common import save_prediction, get_logger
# from simple_net.My_latent import MyLatentModel
from simple_net.Latent import LatentModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7, 8, 9"

parser = argparse.ArgumentParser(description="Training Latent Model")
parser.add_argument("--batch_size", type=int, default=5, help="Training batch size")
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
    # Load dataset
    print('Loading Dataset ...\n')
    data_path = opt.data_path
    dataset_train = Our_Dataset(data_path, data_type='train')
    dataset_test = Our_Dataset(data_path, data_type='test')

    # training data loader
    device_id = [0, 1, 2, 3, 4, 5]
    b_size = len(device_id) * opt.batch_size
    train_loader = DataLoader(dataset=dataset_train, num_workers=32, batch_size=b_size, shuffle=True)
    print('{} training samples'.format(len(dataset_train)))
    print('{} training batches'.format(len(train_loader)))

    # test data loader
    test_loader = DataLoader(dataset=dataset_test, num_workers=32, batch_size=b_size, shuffle=False)
    print('{} test samples'.format(len(dataset_test)))
    print('{} test batches'.format(len(test_loader)))

    # Build model
    net = LatentModel(opt.state_shape, opt.info_shape, opt.action_shape, opt.feature_dim, opt.z1_dim, opt.z2_dim,
                      hidden_units=(256, 256))
    criterion = net.getloss

    # Move to GPU
    model = nn.DataParallel(net.cuda(), device_ids=device_id).cuda()
    # criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    logger = get_logger()
    logger.info('Epoch={}\t lr={}\t'.format(opt.epochs, opt.lr))
    logger.info('Recon Lambda constraint_bound={}\t annealing_rate={}\t'.
                format(net.scheduler_recon.constraint_bound, net.scheduler_recon.annealing_rate))
    logger.info('Classification Lambda constraint_bound={}\t annealing_rate={}\t'.
                format(net.scheduler_classifier.constraint_bound, net.scheduler_classifier.annealing_rate))
    prefix = 'epoch' + str(opt.epochs) + '_lr' + str(opt.lr) \
             + '_Rcb' + str(net.scheduler_recon.constraint_bound) + '_Rar' + str(net.scheduler_recon.annealing_rate) \
             + 'Ccb' + str(net.scheduler_classifier.constraint_bound) + '_Car' + str(net.scheduler_classifier.annealing_rate)

    # training
    running_loss = 0
    train_loss = []
    recon_list = []
    kl_list = []
    classify_list = []
    min_loss = math.inf

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
        if epoch % 50 == 0:
            print('Epoch ' + str(epoch + 1) + ' / ' + str(opt.epochs))
            print('Loss: ' + str(loss))
            print('Loss KL Distance: ' + str(np.mean(loss_kld_list)))
            print('Loss Regression: ' + str(np.mean(loss_recon_list)))
            print('Loss Classification: ' + str(np.mean(loss_classify_list)))

        if loss < min_loss:
            # save model
            model_path = opt.model_path
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), opt.model_path + '/' + prefix + '_slac.pth')

    save_pic = opt.result_path
    plt.plot(train_loss, 'r', label='Total loss')
    plt.title('Total Train Loss')
    plt.savefig(save_pic + prefix + '_trainLoss.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()

    plt.plot(kl_list, 'orange', label='KL Divergence loss')
    plt.title('KL Divergence Loss')
    plt.savefig(save_pic + prefix + '_klLoss.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()

    plt.plot(recon_list, 'g', label='Regression loss')
    plt.title('Regression Loss')
    plt.savefig(save_pic + prefix + '_reconLoss.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()

    # plt.plot(kl_list, 'g-', label='KL Divergence Lambda')
    plt.plot(net.lambda_recon_list, 'g-', label='Regression Lambda')
    # plt.plot(classify_list, 'b-', label='Classification Lambda')
    plt.title('Regression Lambda')
    # plt.legend()
    plt.savefig(save_pic + prefix + '_reconLambda.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()

    plt.plot(classify_list, 'b', label='Classification loss')
    plt.title('Classification Loss')
    plt.savefig(save_pic + prefix + '_classifyLoss.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()

    plt.plot(net.lambda_classify_list, 'b-', label='Classification Lambda')
    plt.title('Classification Lambda')
    # plt.legend()
    plt.savefig(save_pic + prefix + '_classifyLambda.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()

    test_model_path = opt.model_path + '/' + prefix + '_slac.pth'
    test_net = AllRolloutModel(opt.state_shape, opt.info_shape, opt.action_shape, opt.feature_dim, opt.z1_dim,
                               opt.z2_dim, hidden_units=(256, 256))
    test_model = nn.DataParallel(test_net, device_ids=device_id).cuda()
    test_model.load_state_dict(torch.load(test_model_path))
    test_model.eval()

    pred_states = []
    pred_acts = []

    label_states = []
    label_acts = []

    with torch.no_grad():
        test_model.eval()
        for x, act, info, y_all in test_loader:
            x = x.cuda()
            act = act.cuda()
            info = info.cuda()
            # y_all = y_all.cuda()
            pred_state, pred_act = test_model(x, info, act)
            pred_states.append(torch.squeeze(pred_state).cpu().numpy())
            pred_acts.append(torch.squeeze(pred_act).cpu().numpy())

            label_states.append(torch.squeeze(y_all[:, -1, 1:]).cpu().numpy())
            label_acts.append(torch.squeeze(y_all[:, -1, 0:1]).cpu().numpy())

    # classification accuracy
    pred_acts = np.array(pred_acts, dtype=object)
    pred_acts = np.concatenate((pred_acts[0], pred_acts[1]), axis=0)
    pred_acts = np.where(pred_acts > 0.5, 1, 0)
    pred_acts = np.array(pred_acts, dtype=int)
    label_acts = np.array(label_acts, dtype=object)
    label_acts = np.concatenate((label_acts[0], label_acts[1]), axis=0)
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
    pred_states = np.concatenate((pred_states[0], pred_states[1]), axis=0)
    pred_states = np.nan_to_num(pred_states)
    label_states = np.array(label_states, dtype=object)
    label_states = np.concatenate((label_states[0], label_states[1]), axis=0)
    mse = mean_squared_error(label_states, pred_states)
    rmse = np.sqrt(mean_squared_error(label_states, pred_states))
    r2 = r2_score(label_states, pred_states)
    logger.info('auc_score={:.5f}\t mse={:.5f}\t rmse={:.5f}\t r2={:.5f}\t'.
                format(auc_score, mse, rmse, r2))


if __name__ == "__main__":
    main()
