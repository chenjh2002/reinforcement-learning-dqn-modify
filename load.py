import torch
from utils_env import MyEnv
from utils_drl import Agent
import numpy as np

target = 1
model_name = f"model_origin_171"
model_path = f"./save_model/{model_name}"
device = torch.device("cpu")
env = MyEnv(device)
agent = Agent(env.get_action_dim(), device, 0.99, 0, 0, 0, 1, model_path)
print(np.random.randint(10,20))