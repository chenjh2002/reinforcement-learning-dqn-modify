from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import os

def draw():
    # origin part
    # X=[]
    # Y=[]
    # with open("origin_reward.txt",'r') as file:
    #     data=file.readlines()
    #     for row in data:
    #         row=row.rstrip('\n').split(' ')
    #         count=0
    #         for c in row:
    #             if c!='' and count==0:
    #                 iteration=int(c)
    #                 count=1
    #             elif c!='':
    #                 reward=float(c)
    #         X.append(iteration)
    #         Y.append(reward)
    # X=np.array(X)
    # Y=np.array(Y)
    # plt.plot(X,Y,c='orange',label="orgin")
    
    # X=[]
    # Y=[]
    # # ddqn part
    # with open("ddqn_reward.txt",'r') as file:
    #     data=file.readlines()
    #     for row in data:
    #         row=row.rstrip('\n').split(' ')
    #         count=0
    #         for c in row:
    #             if c!='' and count==0:
    #                 iteration=int(c)
    #                 count=1
    #             elif c!='':
    #                 reward=float(c)
    #         X.append(iteration)
    #         Y.append(reward)
    # X=np.array(X)
    # Y=np.array(Y)
    # plt.plot(X,Y,c='red',label="ddqn")
    
    # X=[]
    # Y=[]
    # # priority replay part
    # with open("priority_reward.txt",'r') as file:
    #     data=file.readlines()
    #     for row in data:
    #         row=row.rstrip('\n').split(' ')
    #         count=0
    #         for c in row:
    #             if c!='' and count==0:
    #                 iteration=int(c)
    #                 count=1
    #             elif c!='':
    #                 reward=float(c)
    #         X.append(iteration)
    #         Y.append(reward)
    # X=np.array(X)
    # Y=np.array(Y)
    # plt.plot(X,Y,c='blue',label="priority memory replay")
    
    # best part
    X=[]
    Y=[]
    with open("best_reward.txt",'r') as file:
        data=file.readlines()
        for row in data:
            row=row.rstrip('\n').split(' ')
            count=0
            for c in row:
                if c!='' and count==0:
                    iteration=int(c)
                    count=1
                elif c!='':
                    reward=float(c)
            X.append(iteration)
            Y.append(reward)
    X=np.array(X)
    Y=np.array(Y)
    plt.plot(X,Y,c='orange')
    
    plt.xlabel("iteration times")
    plt.ylabel("average reward")
    plt.title("the best stragtegy reward")
    plt.grid()
    # plt.show()
    plt.savefig(os.path.relpath('./train_curl/best_reward.png'))
draw()