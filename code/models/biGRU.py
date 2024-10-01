import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn

class TextClassificationModel(nn.Module):
    def __init__(self, num_classes, input_size):
        super(TextClassificationModel, self).__init__()
        # Embedding and GRU layers
        self.embed = nn.Embedding(input_size + 1, 200)
        self.gru = nn.GRU(200, 512, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.2)

        # Ensure the input to the linear layer matches the GRU's output
        self.linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, text, offsets):
        offsets = torch.tensor([i for i in offsets if i > 0])
        x = self.embed(text)

        # Pack padded sequences to deal with varying lengths
        packed_input = x
        if (len(offsets)>1):
            packed_input = rnn_utils.pack_padded_sequence(x, offsets.to('cpu'), batch_first=True, enforce_sorted=False)

        # Pass through the GRU
        packed_output, hidden = self.gru(packed_input)

        # Use the hidden state as the output, which has shape [batch_size, hidden_size]
        hidden = hidden[-1]  # Take the last layer's hidden state

        # Apply dropout and pass through the linear layers
        x = self.drop(hidden)
        x = self.linear(x)

        return x
