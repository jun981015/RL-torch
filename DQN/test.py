from utils import *
from models import *
import torch
from matplotlib import animation
import matplotlib.pyplot as plt

def test(num_eval = 10):
    channel = 4
    num_action = 5
    agent = dqn(channel, num_action, 1)
    agent.load_state_dict(torch.load('dqn.pt'))
    agent.eval()
    score,(best_score,frames),q_mean = evaluate(agent=agent,num_eval=num_eval)
    score, best_score = round(score,4),round(best_score,4)
    print(f'average score:{score}, best socre : {best_score}')
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    im = plt.imshow(frames[0])
    def animate(frame):
        plt.clf()
        plt.axis('off')
        plt.imshow(frames[frame])
        plt.title(f'Score {int(best_score)}')
    anim = animation.FuncAnimation(fig, animate, frames=len(frames))
    anim.save('./animation.mp4',fps = 10)
    plt.show()
if __name__ == '__main__':
    test()