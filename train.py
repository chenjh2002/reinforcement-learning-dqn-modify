from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory

from collections import deque
import os
import random
from tqdm import tqdm


'''
basic model
'''
GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./multi_train_models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
MAX_STEPS = 50_000_000
EVALUATE_FREQ = 100_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)

torch.manual_seed(new_seed())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'   # 设置的是通讯的IP地址。在的单机单卡或者单机多卡中，可以设置为'127.0.0.1'（也就是本机）。在多机多卡中可以设置为结点0的IP地址
    os.environ['MASTER_PORT'] = '8888'   # 设置通讯的端口，可以随机设置，只要是空闲端口就可以。

    memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE,"cuda:0")
    mp.spawn(train, nprocs=args.gpus, args=(args,memory,lock))

        
def train(gpu,args,memory,lock):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    env = MyEnv(gpu)
    '''
    set the agent and replaymemory
    '''
    agent = Agent(
        env.get_action_dim(),
        gpu,
        GAMMA,
        new_seed(),
        EPS_START,
        EPS_END,
        EPS_DECAY,
    )
    print("create agent finish")


    torch.cuda.set_device(gpu)
    agent.policy.cuda(gpu)
    agent.target.cuda(gpu)
    # memory.m_states.cuda(gpu)
    # memory.m_actions.cuda(gpu)
    # memory.m_rewards.cuda(gpu)
    # memory.m_dones.cuda(gpu)

    # Wrap the model
    agent.policy = nn.parallel.DistributedDataParallel(agent.policy, device_ids=[gpu])
    agent.target = nn.parallel.DistributedDataParallel(agent.target, device_ids=[gpu])
    
    # Data loading code
    # train_dataset = torchvision.datasets.MNIST(root='./data',
    #                                            train=True,
    #                                            transform=transforms.ToTensor(),
    #                                            download=False)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
    #                                                                 num_replicas=args.world_size,
    #                                                                 rank=rank)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            shuffle=False,
    #                                            num_workers=0,
    #                                            pin_memory=True,
    #                                            sampler=train_sampler)
    if rank==0:
        progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
    else:
        progressive=[i for i in range(MAX_STEPS)]
    
    ###Training
    obs_queue: deque = deque(maxlen=5)
    done = True

    for step in progressive:
        if done:
            observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

        training = len(memory) > WARM_STEPS//4
        state = env.make_state(obs_queue).to(gpu).float()
        action = agent.run(state, training)
        obs, reward, done = env.step(action)
        obs_queue.append(obs)
        
        #共享变量
        memory.push(env.make_folded_state(obs_queue), action, reward, done)

        if step % POLICY_UPDATE == 0 and training:
            memory.m_states= nn.parallel.DataParallel(memory.m_states, device_ids=[gpu])
            memory.m_actions= nn.parallel.DataParallel(memory.m_actions, device_ids=[gpu])
            memory.m_rewards= nn.parallel.DataParallel(memory.m_rewards, device_ids=[gpu])
            memory.m_dones= nn.parallel.DataParallel(memory.m_dones, device_ids=[gpu])

            agent.learn(memory,BATCH_SIZE)

        if step % TARGET_UPDATE == 0:
            agent.sync()

        if step % EVALUATE_FREQ == 0  and rank==0:
            avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
            with open("multi_rewards.txt", "a") as fp:
                fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
            if RENDER:
                prefix = f"eval_{step//EVALUATE_FREQ:03d}"
                os.mkdir(prefix)
                for ind, frame in enumerate(frames):
                    with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                        frame.save(fp, format="png")
            agent.save(os.path.join(
                SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
            done = True
 
 
if __name__ == '__main__':
    main()