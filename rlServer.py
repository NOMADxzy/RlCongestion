from __future__ import division
import torch.nn.functional as F
import copy
from concurrent import futures
import numpy as np
import gc, grpc

import train
import buffer
import StateAndReward_pb2, StateAndReward_pb2_grpc

# env = gym.make('Pendulum-v0')

MAX_EPISODES = 30000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = 1
S_LEN = 8
A_DIM = 9
A_MAX = 1
STATE_LEN = 8

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, S_LEN, A_DIM, A_MAX, ram)


# trainer.load_models(100)

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
        self.rtt_list = []
        self.latency_list = []

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

        return StateAndReward_pb2.Action(action=float(a_idx))

    def update_reward(self, reward, latency, rtt, step):
        self.step_reward.append(reward)
        self.latency_list.append(latency)
        self.rtt_list.append(rtt)

        if step == 0:
            self.avg_acc_reward.append(reward)
        else:
            self.avg_acc_reward.append((step * self.avg_acc_reward[-1] + reward) / (step + 1))


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
