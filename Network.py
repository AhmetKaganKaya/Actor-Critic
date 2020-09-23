import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, fc1, fc2, input, output):
        super(ActorNetwork, self).__init__()
        self.layer1 = nn.Linear(input, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, output)
        self.sm = nn.Softmax(-1)

    def forward(self, observation):
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        output = self.sm(self.layer3(x))
        return output


class CriticNetwork(nn.Module):
    def __init__(self, fc1, fc2, input):
        super(CriticNetwork, self).__init__()
        self.layer1 = nn.Linear(input, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, 1)

    def forward(self, observation):
        x = F.relu(self.layer1(observation))
        x = F.relu(self.layer2(x))
        output = self.layer3(x)
        return output

