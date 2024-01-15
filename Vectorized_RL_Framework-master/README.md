# Vectorized RL Framework
* This framework use envpool==0.8.2 to parallelize rollout, which is compatible with gym==0.26.2.
* Support PPO, DQN (Double and Dueling).

## Installation
```
conda create -n envpool_env python=3.9
conda activate envpool_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install envpool==0.8.2
pip install tensorboard
pip install opencv-python==4.8.0.76
conda install -c conda-forge gym-box2d
```

