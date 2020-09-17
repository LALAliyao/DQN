import gym
import random
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import argparse
import os
import time
import matplotlib.pyplot as plt

#参数声明
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', default=True)
parser.add_argument('--test', dest='test', default=False)

parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=0.1)

parser.add_argument('--train_episodes', type=int, default=200)
parser.add_argument('--test_episodes', type=int, default=10)
args = parser.parse_args()

##创建网络
def crateModel(input_state_shape):
    input_layer=tl.layers.Input(input_state_shape)
    layer1=tl.layers.Dense(32, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None)(input_layer)
    layer2 = tl.layers.Dense(16, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None)(layer1)
    outputlayer = tl.layers.Dense(2, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None)(layer2)
    return tl.models.Model(inputs=input_layer,outputs=outputlayer)
##保存模型
def save_ckpt(model):
    tl.files.save_npz(model.trainable_weights, name='dqn_model.npz')
##加载模型
def load_ckpt(model):
    tl.files.load_and_assign_npz(name="dqn_model.npz",network=model)
##主函数
QNetwork=crateModel([None, 4])
    ##表示QNetwork是训练状态
    QNetwork.train()
    train_weight=QNetwork.trainable_weights
    optimizer=tf.optimizers.SGD(args.lr)
    ##创建环境
    env = gym.make('CartPole-v1')
    if args.train:
        t0=time.time()
        all_episode_reward=[]
        for i in range(args.train_episodes):
            env.render()
            total_reward,done=0,False
            S = env.reset()
            while not done:
                Q=QNetwork(np.asarray([S], dtype=np.float32)).numpy()
                A=np.argmax(Q,1)
                if np.random.rand(1) < args.eps:
                    A[0] = env.action_space.sample()
                S_,reward,done,_=env.step(A[0])
                Q_=QNetwork(np.asarray([S_], dtype=np.float32)).numpy()
                maxQ_=np.max(Q_)
                targetQ=Q
                targetQ[0,A[0]]=reward+0.9*maxQ_
                ##梯度下降更新Q
                with tf.GradientTape() as tape:
                    q_values=QNetwork(np.asarray([S], dtype=np.float32))
                    _loss=tl.cost.mean_squared_error(targetQ, q_values, is_mean=False)
                    grad = tape.gradient(_loss, train_weight)
                optimizer.apply_gradients(zip(grad, train_weight))
                total_reward+=reward
                S=S_
                if done==True:
                    args.eps=1./((i / 50) + 10)
                    break
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
                  .format(i, args.train_episodes, total_reward, time.time() - t0))
            if i==0:
                all_episode_reward.append(total_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + total_reward * 0.1)
        save_ckpt(QNetwork)
