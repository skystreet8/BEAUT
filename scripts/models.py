import torch.nn as nn


class DNNPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: list):
        super(DNNPredictor, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_size)):
            if i == 0:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(input_size, hidden_size[i]),
                        nn.ReLU()
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_size[i - 1], hidden_size[i]),
                        nn.ReLU()
                    )
                )
        self.output_layer = nn.Linear(hidden_size[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
