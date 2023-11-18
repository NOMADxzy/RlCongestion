from __future__ import division
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
S_DIM = 4
A_DIM = 1
A_MAX = 1

print (' State Dimensions :- ', S_DIM)
print (' Action Dimensions :- ', A_DIM)
print (' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
# trainer.load_models(4300)

class RLmethods(StateAndReward_pb2_grpc.acerServiceServicer):

    def __init__(self, max_epi, max_step):
        self.max_epi = max_epi
        self.step = 0
        self.pre_state = []
        self.pre_action = 0
        self.RewardN = 0
        self.hasPre = False
    def GetExplorationAction(self, request, context):
        reward = request.reward
        state = np.float32(request.state)

        if self.hasPre:
            ram.add(self.pre_state, self.pre_action, reward, state)

        action = trainer.get_exploration_action(state)
        self.pre_state = state
        self.pre_action = action
        self.hasPre = True
        self.step += 1

        if ram.len:
            trainer.optimize()

        if self.step % 100 == 0:
            gc.collect()
            print("step:",self.step, "state:",state, "reward:",reward, )
            if self.step %10000 == 0:
                trainer.save_models(self.step/100)

        return StateAndReward_pb2.Action(action=float(action[0]))


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
