# partially inspired from https://sooftware.github.io/kospeech/_modules/kospeech/models/deepspeech2/model.html
import torch
import torch.nn.functional as F
from torch import nn

from hw_asr.base import BaseModel

class BlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

    def forward(self, inputs):
        outputs = F.relu(self.bn(inputs.transpose(1, 2)))
        outputs, _ = self.rnn(outputs.transpose(1, 2))
        return outputs


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, num_rnn_layers=5, rnn_hidden_size=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
            )
        self.layers = nn.ModuleList()
        rnn_output_size = rnn_hidden_size * 2
        rnn_input_size = ((n_feats + 1) // 2 + 1) // 2 * 32

        for i in range(num_rnn_layers):
            self.layers.append(BlockRNN(
                input_size=rnn_input_size if i == 0 else rnn_output_size,
                hidden_size=rnn_hidden_size
            ))

        self.fc = nn.Sequential(
            nn.LayerNorm(rnn_output_size),
            nn.Linear(rnn_output_size, n_class, bias=False),
        )

    def forward(self, spectrogram, *args, **kwargs):
        inputs = spectrogram.unsqueeze(1)
        outputs = self.conv(inputs)
        batch_size, num_channels, hidden_dim, seq_length = outputs.size()
        outputs = outputs.view(batch_size, num_channels * hidden_dim, seq_length).permute(2, 0, 1)

        for layer in self.layers:
            outputs = layer(outputs)

        outputs = outputs.permute(1, 0, 2)
        return self.fc(outputs)

    def transform_input_lengths(self, input_lengths):
        return (input_lengths + 1) // 2