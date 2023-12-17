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

from LSTM.custom_LSTM import LSTM
from Our_Dataset import Our_Dataset
from common import save_prediction, get_logger
# from simple_net.My_latent import MyLatentModel
from simple_net.Latent import LatentModel
from simple_net.My_latent import MyLatentModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7, 8, 9"

parser = argparse.ArgumentParser(description="Training LSTM Model")
parser.add_argument("--batch_size", type=int, default=3, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")

parser.add_argument("--state_shape", type=tuple, default=(12,))
parser.add_argument("--info_shape", type=tuple, default=(4,))
parser.add_argument("--action_shape", type=tuple, default=(1,))
parser.add_argument("--num_sequence", type=int, default=5)

parser.add_argument("--data_path", type=str, default='/home/fan/NtChen/MyData/')
parser.add_argument("--result_path", type=str, default='/home/fan/NtChen/Results/')
parser.add_argument("--model_path", type=str, default='/home/fan/NtChen/Models/')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading Dataset ...\n')
    data_path = opt.data_path
    dataset_train = Our_Dataset(data_path, data_type='train')
    dataset_test = Our_Dataset(data_path, data_type='test')

    # training data loader
    device_id = [0, 1, 2, 3]
    b_size = len(device_id) * opt.batch_size
    train_loader = DataLoader(dataset=dataset_train, num_workers=32, batch_size=b_size, shuffle=True)
    print('{} training samples'.format(len(dataset_train)))
    print('{} training batches'.format(len(train_loader)))

    # test data loader
    test_loader = DataLoader(dataset=dataset_test, num_workers=32, batch_size=b_size, shuffle=False)
    print('{} test samples'.format(len(dataset_test)))
    print('{} test batches'.format(len(test_loader)))

    # Build model
    net = LSTM(info_dim=4, input_dim=13, hidden_dim=128, output_dim=13, num_layers=2)
    criterion = net.getloss

    # Move to GPU
    model = nn.DataParallel(net.cuda(), device_ids=device_id).cuda()
    # criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    logger = get_logger()
    logger.info('Epoch={}\t lr={}\t'.format(opt.epochs, opt.lr))
    prefix = 'epoch' + str(opt.epochs) + '_lr' + str(opt.lr) + '_Rcb10' + '_Ccb1'

    # training
    running_loss = 0
    train_loss = []
    recon_list = []
    classify_list = []
    min_loss = math.inf

    for epoch in tqdm(range(opt.epochs)):
        model.train()
        loss_list = []
        loss_recon_list = []
        loss_classify_list = []
        for x, acts, info, y_gt in train_loader:
            # training step
            model.zero_grad()
            optimizer.zero_grad()

            # [B, 5, 13]
            x = x.cuda()

            # [B, 5, 13] acts+y_states
            y_gt = y_gt.cuda()

            # [B, 4, 1]
            info = info.cuda()

            acts = acts.cuda()

            y = model(x, info, acts)

            loss_recon, loss_classification, loss = criterion(y, y_gt)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            loss_recon_list.append(loss_recon.item())
            loss_classify_list.append(loss_classification.item())

        # at the end of each epoch
        loss = np.mean(loss_list)
        running_loss += loss
        train_loss.append(loss)
        recon_list.append(np.mean(loss_recon_list))
        classify_list.append(np.mean(loss_classify_list))
        if epoch % 100 == 0:
            print('Epoch ' + str(epoch + 1) + ' / ' + str(opt.epochs))
            print('Loss: ' + str(loss))
            print('Loss Regression: ' + str(np.mean(loss_recon_list)))
            print('Loss Classification: ' + str(np.mean(loss_classify_list)))

        if loss < min_loss:
            min_loss = loss
            # save model
            model_path = opt.model_path
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), opt.model_path + '/' + prefix + '_LSTM.pth')

    save_pic = opt.result_path
    plt.plot(train_loss, 'r', label='Total loss')
    plt.title('Total Train Loss')
    plt.savefig(save_pic + prefix + '_trainLoss.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()

    plt.plot(recon_list, 'g', label='Regression loss')
    plt.title('Regression Loss')
    plt.savefig(save_pic + prefix + '_reconLoss.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()
    #
    plt.plot(classify_list, 'b', label='Classification loss')
    plt.title('Classification Loss')
    plt.savefig(save_pic + prefix + '_classifyLoss.png', bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()
    #
    # plt.plot(net.lambda_classify_list, 'b-', label='Classification Lambda')
    # plt.title('Classification Lambda')
    # # plt.legend()
    # plt.savefig(save_pic + prefix + '_classifyLambda.png', bbox_inches='tight', pad_inches=0.0)
    # plt.show()
    # plt.close()


    test_model_path = opt.model_path + prefix + '_LSTM.pth'
    test_net = LSTM(info_dim=4, input_dim=13, hidden_dim=128, output_dim=13, num_layers=2)
    Rollout = test_net.rollouts
    test_model = nn.DataParallel(test_net, device_ids=device_id).cuda()
    test_model.load_state_dict(torch.load(test_model_path))
    test_model.eval()

    pred_states = []
    pred_acts = []

    label_states = []
    label_acts = []

    with torch.no_grad():
        test_model.eval()
        for x, acts, info, y_gt in test_loader:
            x = x.cuda()
            info = info.cuda()
            acts = acts.cuda()
            y_gt = y_gt.cuda()
            pred = Rollout(x, info, acts, y_gt)
            # pred = test_model(x, info, acts)
            pred_states.append(torch.squeeze(pred[:, -1, 1:]).cpu().numpy())
            pred_acts.append(torch.squeeze(pred[:, -2, 0]).cpu().numpy())

            label_states.append(torch.squeeze(y_gt[:, -1, 1:]).cpu().numpy())
            label_acts.append(torch.squeeze(y_gt[:, -2, 0:1]).cpu().numpy())

    # classification accuracy
    pred_acts = np.array(pred_acts, dtype=object)
    pred_acts = np.concatenate((pred_acts[0], pred_acts[1], pred_acts[2], pred_acts[3], pred_acts[4]), axis=0)
    pred_acts = np.where(pred_acts > 0.5, 1, 0)
    pred_acts = np.array(pred_acts, dtype=int)
    label_acts = np.array(label_acts, dtype=object)
    label_acts = np.concatenate((label_acts[0], label_acts[1], label_acts[2], label_acts[3], label_acts[4]), axis=0)
    label_acts = np.array(label_acts, dtype=int)

    fpr, tpr, thresholds = roc_curve(label_acts, pred_acts)
    auc_score = roc_auc_score(label_acts, pred_acts)
    # gmeans = math.sqrt(tpr * (1 - fpr))
    # # Find the optimal threshold
    # index = max(gmeans)
    # optimal_threshold = thresholds[index]
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')  # 设置标题
    plt.legend()
    plt.show()

    # regression accuracy
    pred_states = np.array(pred_states, dtype=object)
    pred_states = np.concatenate((pred_states[0], pred_states[1], pred_states[2], pred_states[3], pred_states[4]), axis=0)
    pred_states = np.nan_to_num(pred_states)
    label_states = np.array(label_states, dtype=object)
    label_states = np.concatenate((label_states[0], label_states[1], label_states[2], label_states[3], label_states[4]), axis=0)
    mse = mean_squared_error(label_states, pred_states)
    rmse = np.sqrt(mean_squared_error(label_states, pred_states))
    r2 = r2_score(label_states, pred_states)
    logger.info('auc_score={:.5f}\t mse={:.5f}\t rmse={:.5f}\t r2={:.5f}\t'.
                format(auc_score, mse, rmse, r2))


if __name__ == "__main__":
    main()
