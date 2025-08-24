import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2,output_dim=310):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    def forward(self, input, hidden, cell, time_steps):
        out =[]
        out1 = self.fc_out(input)
        out.append(out1)
        out1= out1.unsqueeze(0)  # input = [batch size] to [1, batch size]
        for i in range(time_steps-1):
            output, (hidden, cell) = self.rnn(out1,
                                              (hidden, cell))  # output =[seq len, batch size, hid dim* ndirection]
            out_cur = self.fc_out(output.squeeze(0))  # prediction = [batch size, output dim]
            out.append(out_cur)
            out1 = out_cur.unsqueeze(0)
        out.reverse()
        out = torch.stack(out)
        out = out.transpose(1,0) #batch first
        return out, hidden, cell