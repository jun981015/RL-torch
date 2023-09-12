import numpy as np
from gym import Wrapper,make
from cv2 import cvtColor,COLOR_RGB2GRAY
import random,os,torch
def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
def evaluate(agent, num_eval=3):
    env = make('CarRacing-v2', continuous=False,render_mode = 'rgb_array')
    env = CarRacingEnv(env)
    scores = 0
    best_score = - np.inf
    agent.eval()
    states ,q_val= [],[]
    done = False
    for _ in range(num_eval):
        s = env.reset()
        R = 0
        S = []
        while not done:
            S.append(env.render())
            a,q = agent.select_action(s)
            q_val.append(q)
            ns, r, terminated, truncated = env.step(a)
            s = ns
            R += r
            done = terminated or truncated
        if R > best_score:
            best_score = R
            states = S[:]
        scores += R
        done = False
    return round(scores / num_eval,4),(best_score, states),np.mean(q_val)
class CarRacingEnv(Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, skip_start=50):
        super(CarRacingEnv, self).__init__(env,)
        self.env = env
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.skip_start = skip_start
    def __img_process(self,img):
        return cvtColor(img[:84, 6:90], COLOR_RGB2GRAY) / 255.0
    def reset(self):
        observation = self.env.reset()
        for _ in range(self.skip_start): observation = self.env.step(0)
        s = self.__img_process(observation[0])
        self.state = np.stack([s]*self.stack_frames)  
        return self.state
    def step(self, action):
        R = 0
        for _ in range(self.skip_frames): 
            observation = self.env.step(action)
            s,r,terminated ,truncated = observation[:4]
            R += r
            if terminated or truncated:
                break
        s = self.__img_process(s)
        self.state[:3] = self.state[1:]
        self.state[3:] = s[np.newaxis]
        return (self.state, R, terminated, truncated)