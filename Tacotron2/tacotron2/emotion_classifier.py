import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class EmotionClassifier(nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        # self.layers = nn.Sequential(
        #     nn.Linear(emb_size, 512),
        #     torch.nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     torch.nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     torch.nn.ReLU(),
        #     nn.Linear(1024, 5)
        # )

        # self.layers = nn.Sequential(
        #     nn.Linear(emb_size, 512),
        #     torch.nn.Tanh(),
        #     nn.Linear(512, 1024),
        #     torch.nn.Tanh(),
        #     nn.Linear(1024, 1024),
        #     torch.nn.Tanh(),
        #     nn.Linear(1024, 5)
        # )

        self.layers = nn.Sequential(
            nn.Linear(emb_size, 512),
            torch.nn.ReLU(),
            # nn.Linear(512, 1024),
            # torch.nn.LeakyReLU(),
            # nn.Linear(1024, 1024),
            # torch.nn.LeakyReLU(),
            nn.Linear(512, 5)
        )

    def forward(self, inputs):
        out = self.layers(inputs)
        return out