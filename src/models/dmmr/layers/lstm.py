import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim=310, output_dim=64, layers=2, location=-1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=layers, batch_first=True)
        self.location = location
    def forward(self, x):
        # self.lstm.flatten_parameters()
        feature, (hn, cn) = self.lstm(x)
        return feature[:, self.location, :], hn, cn