import numpy as np
import torch
import torch.nn as nn

class ActorNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, N_frame: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(N_frame, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down3 = nn.MaxPool2d(kernel_size=2)

        self.linear = nn.Sequential(
            nn.Linear(16*(state_dim//8)**2, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh()
        )

    def forward(self, state, sigma=0.0, brake_rate=0.015):
        # state = state.float() / 255.0
        h = self.conv1(state)
        h = self.down1(h)
        h = self.conv2(h)
        h = self.down2(h)
        h = self.conv3(h)
        h = self.down3(h)
        h = torch.flatten(h, start_dim=1)
        h = self.linear(h)

        # add normal distribution noise to action (if you add noise here, you don't need to add noise in decide_agent_actions)
        # h_clone = h.clone() + torch.randn_like(h) * sigma
        h_clone = h.clone()
        # map to valid action space: {steer:[-1, 1], gas:[0, 1], brake:[0, 1]}
        h_clone[:, 0] = (h_clone[:, 0])
        h_clone[:, 1] = (h_clone[:, 1]+1) * 0.5 + 0.1
        h_clone[:, 2] = (h_clone[:, 2]+1) * brake_rate

        
        return h_clone


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, N_frame: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(N_frame, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
        )
        self.down1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish()
        )
        self.down3 = nn.MaxPool2d(kernel_size=2)

        self.action_linear = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LayerNorm(256),
            nn.Mish()
        )

        self.state_linear = nn.Sequential(
            nn.Linear(16*(state_dim//8)**2, 256),
            nn.LayerNorm(256),
            nn.Mish(),
        )

        self.concat_linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        # extract the state features
        # state = state.float() / 255.0
        state_h = self.conv1(state)
        state_h = self.down1(state_h)
        state_h = self.conv2(state_h)
        state_h = self.down2(state_h)
        state_h = self.conv3(state_h)
        state_h = self.down3(state_h)
        state_h = torch.flatten(state_h, start_dim=1)

        state_h = self.state_linear(state_h)
        # action features
        action_h = self.action_linear(action)

        # concat
        h = self.concat_linear(torch.concat((state_h, action_h), dim=1))

        return h
    
class ActorAtariNet(nn.Module):
    def __init__(self, action_dim, N_frame, init_weights=True):
        super(ActorAtariNet, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(N_frame, 16, kernel_size=5, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(16, 32, kernel_size=3, stride=3),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 32, kernel_size=3, stride=3),
                                        nn.ReLU(True)
                                        )
        self.linear = nn.Sequential(nn.Linear(128, 64),
                                        nn.ReLU(True),
                                        nn.Linear(64, action_dim),
                                        nn.Tanh()
                                        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x
        
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

class CriticAtariNet(nn.Module):
    def __init__(self, action_dim, N_frame, init_weights=True):
        super(CriticAtariNet, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(N_frame, 16, kernel_size=5, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(16, 32, kernel_size=3, stride=3),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 32, kernel_size=3, stride=3),
                                        nn.ReLU(True)
                                        )
        self.linear = nn.Sequential(nn.Linear(128, 64),
                                        nn.ReLU(True)
                                        )
        
        self.concat_linear = nn.Sequential(nn.Linear(64 + action_dim, 64),
                                            nn.ReLU(True),
                                            nn.Linear(64, 32),
                                            nn.ReLU(True),
                                            nn.Linear(32, 1)
                                            )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, action):
        # x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        h = self.concat_linear(torch.concat((x, action), dim=1))

        return h
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)