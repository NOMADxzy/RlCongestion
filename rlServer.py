from __future__ import division
import torch.nn.functional as F
import copy
from concurrent import futures
import numpy as np
import gc, grpc

import train
import buffer
import grpc_pb.StateAndReward_pb2 as StateAndReward_pb2
import grpc_pb.StateAndReward_pb2_grpc as StateAndReward_pb2_grpc
import matplotlib.pyplot as plt

# env = gym.make('Pendulum-v0')

MAX_EPISODES = 30000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = 1
S_LEN = 8
A_DIM = 13
A_MAX = 1
STATE_LEN = 8

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, S_LEN, A_DIM, A_MAX, ram)


# trainer.load_models(2800,s=1,v=15)

class RLmethods(StateAndReward_pb2_grpc.acerServiceServicer):

    def __init__(self, max_epi, max_step):
        # 超参数
        self.max_epi = max_epi
        self.step = 0
        # 前若干个时间序列存储
        self.state_hist = []
        self.pre_state = []
        self.pre_action = 0
        # 绘图
        self.step_reward = []
        self.avg_acc_reward = []
        self.throughput_list = []
        self.latency_list = []
        self.plot_step = 0
        self.plt_interval = 1

    def GetExplorationAction(self, request, context):
        reward = request.reward
        state = np.float32(request.state)

        if len(self.state_hist) == STATE_LEN:
            pre_state = copy.copy(self.state_hist)
            ram.add(pre_state, self.pre_action, reward, state)

        self.state_hist.append(state)
        while len(self.state_hist) > STATE_LEN:
            self.state_hist = self.state_hist[1:]

        while len(self.state_hist) < STATE_LEN:
            self.state_hist.insert(0, self.state_hist[0])

        action, a_idx = trainer.get_exploration_action((np.array(self.state_hist)))
        self.pre_action = action
        self.step += 1

        if ram.len:
            trainer.optimize()

        if self.step % 100 == 0:
            gc.collect()
            print("step:", self.step, "state:", state, "reward:", reward, )
            if self.step % 10000 == 0:
                trainer.save_models(int(self.step / 100))

        return StateAndReward_pb2.Action(action=a_idx, action_dim=A_DIM // 2)

    def UpdateMetric(self, request, context):
        metrics = request.metrics
        self.update_reward(reward=metrics[0], latency=metrics[1], throughput=metrics[2])
        print("got metrics :", metrics)
        return StateAndReward_pb2.Res(r=float(self.plot_step))

    def update_reward(self, reward, latency, throughput):
        self.step_reward.append(reward)
        self.latency_list.append(latency)
        self.throughput_list.append(throughput)

        if self.plot_step == 0:
            self.avg_acc_reward.append(reward)
        else:
            self.avg_acc_reward.append((self.plot_step * self.avg_acc_reward[-1] + reward) / (self.plot_step + 1))
        self.plot_step += 1
        if self.plot_step % 1000 == 0:
            self.plot_all()

    def plot_all(self):
        self.sample_plot(self.step_reward, self.plt_interval, "step_reward")
        self.sample_plot(self.avg_acc_reward, self.plt_interval, "avg_acc_reward")
        self.sample_plot(self.throughput_list, self.plt_interval, "throughput")
        self.sample_plot(self.latency_list, self.plt_interval, "latency")

    def sample_plot(self, vals, interval, msg):  # 按interval采样vals并绘图
        plt.figure()
        new_vals = self.resize(vals, interval)
        num_of_x = len(new_vals)
        plt.plot([i * interval for i in range(num_of_x)],
                 new_vals, label=msg)
        plt.legend()
        plt.xlabel("step")
        plt.ylabel(msg)
        plt.savefig('./results/' + msg + '.png')

    def resize(self, val_list, a):
        new_list = []
        for idx, e in enumerate(val_list):
            if idx % a == 0:
                new_list.append(e)
        return new_list


if __name__ == '__main__':
    # 实例化 server 线程池数量为10
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # Producter 注册逻辑到 server 中
    StateAndReward_pb2_grpc.add_acerServiceServicer_to_server(RLmethods(MAX_EPISODES, MAX_STEPS), server)
    # 启动 server
    server.add_insecure_port('[::]:50053')
    server.start()
    print("rpc serve in port 50053")
    server.wait_for_termination()
