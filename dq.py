# adapted from https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/DeepQLearning
import time
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, alpha, num_pos, fc1_dims, fc2_dims,
                 n_actions, q):
        super(DeepQNetwork, self).__init__()
        self.q = q
        self.num_pos = num_pos
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.fc1 = nn.Linear(self.num_pos * (self.q + 1), self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        # check_nan(observation)
        state = T.Tensor(observation).to(self.device)
        state = state.to(T.int64)
        x = nn.functional.one_hot(state, self.q + 1)
        x = x.to(T.float)
        # print("unflattened", x.shape)
        # print(x)
        if len(x.shape) == 3:  # if we are processing a batch
            x = x.flatten(start_dim=1)
        else:
            x = x.flatten()
        # print("x_shape", x.shape)
        # print(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        norm_actions = actions
        return norm_actions


class Agent(object):
    def __init__(self, gamma, epsilon, alpha, num_pos, batch_size, n_actions, q,
                 max_mem_size=100000, eps_end=0.01, eps_dec=0.996):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_MIN = eps_end
        self.EPS_DEC = eps_dec
        self.ALPHA = alpha
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.zero_long = T.zeros([1], dtype=T.long, device='cuda')
        self.Q_eval = DeepQNetwork(alpha, n_actions=self.n_actions,
                                   num_pos=num_pos, fc1_dims=2048, fc2_dims=2048, q=q)
        self.state_memory = np.zeros((self.mem_size, num_pos))
        self.new_state_memory = np.zeros((self.mem_size, num_pos))
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def storeTransition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - terminal
        self.mem_cntr += 1

    def chooseAction(self, observation, probabilistic=True):
        rand = np.random.random()
        if rand > self.EPSILON:
            actions = self.Q_eval.forward(observation)
            taken = observation != 0
            actions = actions.masked_fill(T.tensor(taken, device=self.Q_eval.device), -np.inf)
            action = T.argmax(actions).item()
        else:
            options = np.where(observation == 0)[0]
            action = np.random.choice(options)  # choose a random empty cell
        return action

    def chooseAction_probabilistic(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand > self.EPSILON:
            action = (actions.cumsum(0) > np.random.random()).nonzero().take(
                self.zero_long).item()  # choose probabilistically
            taken = observation == 0  # todo: apply to probabilistic move choice
            actions = actions.masked_fill(T.tensor(taken, device=self.Q_eval.device), -np.inf)
            action = T.argmax(actions).item()
        else:
            options = np.where(observation == 0)[0]
            action = np.random.choice(options)  # choose a random empty cell
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()
            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size

            batch = np.random.choice(max_mem, self.batch_size)
            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.uint8)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            terminal_batch = self.terminal_memory[batch]

            reward_batch = T.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = T.Tensor(terminal_batch).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = q_eval.clone()
            q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            action_indices = [int(action_indices[i]) for i in range(len(action_indices))]  # todo: WHY????
            q_target[batch_index, action_indices] = reward_batch + self.GAMMA * T.max(q_next, dim=1)[0] * terminal_batch
            self.EPSILON = self.EPSILON * self.EPS_DEC if self.EPSILON > self.EPS_MIN else self.EPS_MIN
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()

    def save_model(self, path):
        T.save(self.Q_eval.state_dict(), path)
        print("Saved model to", path)

#
# alpha = 0.003
# num_pos = 3 ** 2
# q = 2
# model = DeepQNetwork(alpha, n_actions=num_pos,
#                      num_pos=num_pos, fc1_dims=2048, fc2_dims=2048, q=q)
# model.load_state_dict(T.load('models/5k_one_hot_player_1.pth'))
# model.eval()
# b = Board.blank_board(3, 2, 2)
# b.cli()
# fwded = model.forward(b.to_linear_array())
# actions = fwded / fwded.sum()
# print(actions)
# cumprobs = actions.cumsum(0)
# print(cumprobs)
# reps = int(1e3)
# start = time.time()
# z = T.tensor([0], device='cuda')
# l = []
# for _ in range(reps):
#     r = np.random.random()
#     a = (cumprobs > r).nonzero().take(z).item()
#     l.append(a)
# end = time.time()
# print(end- start)
# for i in range(9):
#     print("occurences of", i, ":", l.count(i), fwded[i])
