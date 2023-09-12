from utils import *
from models import *
import gym
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
NUM_FRAME = int(3e5)
EPS_MAX = 0.6
EPS_MIN = 0.1
FINAL_EXPLORATION = 1e5
BATCH_SIZE = 64
START_TRAIN = BATCH_SIZE*50
BUFFER_SIZE = 20000
GAMMA = 0.99
LR = 0.00025
TAU = 0.25
target_interval = 500
eval_interval = target_interval
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train():
    env = gym.make('CarRacing-v2', continuous=False)
    env = CarRacingEnv(env)
    channel = 4
    num_action = env.action_space.n
    agent = dqn(channel, num_action, BUFFER_SIZE)
    agent.to(device)
    Loss = nn.MSELoss()
    optim = torch.optim.Adam(agent.Q.parameters(),LR)
    epsilon = EPS_MAX
    s=env.reset()
    best_score =  - np.inf
    num_ep ,f =0, 0
    scores , steps, q_eval= [],[],[]
    r_sum,loss_sum ,q_sum= 0 ,0 ,0
    for i in range(NUM_FRAME):
        a,action_value = agent.select_action(s,epsilon)
        f+=1
        ns, r, terminated, truncated = env.step(a)
        q_sum +=action_value
        r_sum +=r
        agent.replay_buffer.append((s,ns,a,r,terminated))
        s = ns
        if START_TRAIN < i:
            q_value ,td_target = agent.get_error(BATCH_SIZE,GAMMA)
            loss = Loss(q_value ,td_target)
            loss_sum += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
        if terminated or truncated:
            s = env.reset()
            num_ep+=1
            print(f"{i+1} frame, {num_ep} episode, total reward : {round(r_sum,4)}, avg q {round(q_sum/f,4)}, avg loss {round(loss_sum/f,4)}")
            r_sum,loss_sum, q_sum,f = 0,0,0,0
        if (i+1)%target_interval == 0: agent.target_update(tau = TAU)
        if i < FINAL_EXPLORATION: epsilon -= (EPS_MAX-EPS_MIN)/FINAL_EXPLORATION
        if (i+1)%eval_interval == 0:
            score,_,q_e = evaluate(agent=agent,num_eval=5)
            agent.train()
            if score > best_score:
                best_score = score
                torch.save(agent.state_dict(), 'my_dqn.pt')
            print(f'{i+1} frame, eval score : {score}, eps : {round(epsilon,4)}, avg q {round(q_e)}, best score : {best_score},  ')
            scores.append(score)
            steps.append(i+1)
            q_eval.append(q_e)
            if (i+1)%(eval_interval*8) == 0:
                fig, ax1 = plt.subplots()
                ax1.plot(steps,scores, 'r-',label = 'scores')
                ax2 = ax1.twinx()
                ax2.plot(steps,q_eval,'b-',label = 'Q value')
                ax1.set_xlabel('Step', fontsize=14)
                ax1.set_ylabel('AvgReturn', fontsize=14)
                ax2.set_ylabel('AvgQvalue',fontsize = 14)
                ax1.grid(True,axis='both')
                fig.legend()
                plt.show(block=False)
                plt.pause(5)
                plt.close()
    print("done")
if __name__ =="__main__":
    train()