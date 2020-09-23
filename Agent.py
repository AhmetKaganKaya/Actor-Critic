from Network import ActorNetwork, CriticNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Agent():
    def __init__(self, gamma, n_actions, input_dims):
        self.gamma = gamma
        self.n_actions = n_actions
        self.input_dims = input_dims

        # fc1 ve fc2 değerleri değişebilir
        self.critic_network = CriticNetwork(fc1=256, fc2=256, input=self.input_dims).double()
        self.actor_network = ActorNetwork(fc1=256, fc2=256, input=self.input_dims, output = self.n_actions).double()

        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=0.00005)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=0.00005)

    def choose_action(self, observation):
        # Choose action according to policy
        state = torch.tensor(observation).double()
        probability = self.actor_network.forward(state)
        dist = torch.distributions.Categorical(probs=probability)
        action = dist.sample()

        distributon = torch.log(probability)
        return action.item(), distributon

    def learn(self, state, action, next_state, reward, done, distribution):

        # Advantage function calculation
        q_eval = self.critic_network.forward(torch.tensor(state).double())
        q_next = self.critic_network.forward(torch.tensor(next_state).double()).detach()

        advantage = reward + self.gamma * (1 - int(done)) * q_next - q_eval
        print("Advantage: {}".format(advantage))
        # Critic Loss Calculation
        self.critic_optimizer.zero_grad()
        loss_cr = advantage.pow(2)
        loss_cr.backward()
        self.critic_optimizer.step()

        # Actor Loss Calculation
        self.actor_optimizer.zero_grad()
        policy_loss = -(distribution[action] * advantage.detach())
        policy_loss.backward()
        self.actor_optimizer.step()
