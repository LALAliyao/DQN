import time
import gym
import numpy as np
from gym_demo import load_ckpt,crateModel

##创建环境
env = gym.make('CartPole-v1')

t0 = time.time()
##与模型建立一个相同的网络从而正确加载网络参数
QNetwork_test=crateModel([None, 4])
load_ckpt(QNetwork_test)

num_episodes=300

##表示QNetwork是评估状态，必须要写
QNetwork_test.eval()

##开始评估
for i in range(num_episodes):
    episode_time = time.time()
    S=env.reset()
    total_reword,done=0,False
    env.render()
    step_counter=0
    while not done:
        step_counter += 1
        Q=QNetwork_test(np.asarray([S],np.float32)).numpy()
        A=np.argmax(Q,1)
        S_,reward,done,_=env.step(A[0])
        total_reword+=reward
        S=S_
        if done==True:
            print(step_counter)
            break

    print('Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
          .format(i, num_episodes, total_reword, time.time() - t0))
