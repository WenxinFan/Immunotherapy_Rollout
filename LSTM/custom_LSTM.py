import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.autograd import Variable


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, info_dim, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.info_dim = info_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = 5

        self.info_layer = nn.Linear(info_dim, int(hidden_dim/2)-input_dim)
        self.fuse_layer = nn.Linear(int(hidden_dim/2), hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.single_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, 1, batch_first=True)
        self.leakyReLU = nn.LeakyReLU(0.2, inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_states, info, acts):
        # x_states.shape = (B, 5, 12)
        # Initialize hidden state and cell state
        x = torch.cat((acts, x_states), dim=-1).cuda()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()

        initial_data = torch.permute(x[:, 0:1, :], [0, 2, 1])

        # Custom initialization for the first step, if initial_data is provided
        if initial_data is not None:
            # Process initial_data to get the initial hidden state and cell state
            custom_h0, custom_c0 = self.initialize_first_step(initial_data[:, :, 0], info[:, :, 0])
            h0[0, :custom_h0.size(0), :custom_h0.size(1)] = custom_h0
            c0[0, :custom_h0.size(0), :custom_c0.size(1)] = custom_c0

        # Forward propagate the LSTM
        output, _ = self.lstm(x, (h0, c0))
        output = self.fc(output)
        output_acts = self.sigmoid(output[:, :, 0])
        output[:, :, 0] = output_acts
        return output

    def initialize_first_step(self, initial_data, info):
        # Custom function to initialize h0 and c0 based on initial_data
        # For example, using a linear layer to transform initial_data to the hidden size

        info_out = self.leakyReLU(self.info_layer(info))
        # Concatenate individual information with initial state
        fused_input = torch.cat((info_out, initial_data), dim=1)
        temp = self.fuse_layer(fused_input)
        fused_out = self.leakyReLU(temp)
        return fused_out, fused_out

    def getloss(self, pred, label):
        # regression loss
        loss1 = torch.mean(torch.pow((pred[:, -1, 1:] - label[:, -1, 1:]), 2))
        loss_recon = loss1 / torch.mean(torch.pow(label[:, -1, 1:], 2))

        # classification loss
        binary_loss = nn.BCEWithLogitsLoss()
        loss_classification = binary_loss(pred[:, -2, 0], label[:, -2, 0])

        loss = (
                10 * loss_recon
                + 1 * loss_classification
        )

        return loss_recon, loss_classification, loss

    def rollouts(self, x_states, info, acts, y_all):
        """
        state.shape = (B, 5, 12)
        info.shape = (B, 4, 1)
        action.shape = (B, 5, 1)
        """
        n_init_steps = 2  # TODO: set in the config file
        x = torch.cat((acts, x_states), dim=-1).cuda()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()

        initial_data = torch.permute(x[:, 0:1, :], [0, 2, 1])

        # Custom initialization for the first step, if initial_data is provided
        if initial_data is not None:
            # Process initial_data to get the initial hidden state and cell state
            custom_h0, custom_c0 = self.initialize_first_step(initial_data[:, :, 0], info[:, :, 0])
            h0[0, :custom_h0.size(0), :custom_h0.size(1)] = custom_h0
            c0[0, :custom_h0.size(0), :custom_c0.size(1)] = custom_c0

        # Forward propagate the LSTM
        # output (B, seq)
        output, hc = self.lstm(x[:, :n_init_steps, :], (h0, c0))
        h = hc[0][0:1]
        c = hc[1][0:1]
        hc = (h, c)

        for t in range(n_init_steps, self.sequence_length):
            output, hc = self.single_lstm(output, hc)

        output = self.fc(output)
        pred_acts = self.sigmoid(output[:, :, 0])
        output[:, :, 0] = pred_acts
        # pred_states = output[:, -1, 1:]

        # TODO: now we only compute the r2 of the last output, but probably we need to compute the r2 of all outputs
        # with torch.no_grad():
        #
        #     pred_states = np.squeeze(output[:, -1, 1:].cpu().detach().numpy())
        #     label_states = y_all[:, -1, 1:].cpu().detach().numpy()
        #     mse = mean_squared_error(label_states, pred_states)
        #     rmse = np.sqrt(mean_squared_error(label_states, pred_states))
        #     r2 = r2_score(label_states, pred_states)
        #
        # # last_state, last_act
        # outputs = {"rollout_r2": torch.tensor(r2, dtype=torch.float32, requires_grad=False),
        #            "rollout_rmse": torch.tensor(rmse, dtype=torch.float32, requires_grad=False)}

        return output

