import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random


class AtariDQNAgent(DQNBaseAgent):
    def __init__(self, config):
        super(AtariDQNAgent, self).__init__(config)
        ### TODO ###
        # initialize env
        # set a atari game
        env = gym.make(config["env_id"], render_mode='rgb_array')
        # resize screen to 84x84
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        # convert to grayscale
        env = gym.wrappers.GrayScaleObservation(env)
        # stack 4 frames
        env = gym.wrappers.FrameStack(env, 4)
        self.env = env

        ### TODO ###
        # initialize test_env
        self.test_env = env

        # initialize behavior network and target network
        self.behavior_net = AtariNetDQN(self.env.action_space.n)
        self.behavior_net.to(self.device)
        self.target_net = AtariNetDQN(self.env.action_space.n)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        # initialize optimizer
        self.lr = config["learning_rate"]
        self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

    def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
        ### TODO ###
        # get action from behavior net, with epsilon-greedy selection
        observation = torch.from_numpy(observation.__array__())
        observation = observation.to(self.device, dtype=torch.float32)

        if random.random() < epsilon:
            action = np.random.randint(0, action_space.n, size=1).item()
        else:
            action = self.behavior_net(observation.unsqueeze(0)).argmax(dim=1).cpu().numpy().item()

        return action

        return NotImplementedError

    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

        ### TODO ###
        # calculate the loss and update the behavior network
        # 1. get Q(s,a) from behavior net
        # 2. get max_a Q(s',a) from target net
        # 3. calculate Q_target = r + gamma * max_a Q(s',a)
        # 4. calculate loss between Q(s,a) and Q_target
        # 5. update behavior net

        action = action.type(torch.long)
        q_value = self.behavior_net(state).gather(1, action)
        with torch.no_grad():
            q_next = self.behavior_net(next_state)
            action_index = q_next.max(dim=1)[1].view(-1, 1)
            # choose related Q from target net
            q_next = self.target_net(next_state).gather(dim=1, index=action_index.long())
            # if episode terminates at next_state, then q_target = reward
            q_target = reward + self.gamma * q_next * (1 - done)

        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)

        self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

