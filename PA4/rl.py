from __future__ import division
import numpy as np
from gridworld import GridworldEnv
import sys


class RL(object):

    def __init__(self, grid, alpha=0.01, gamma=0.99, l=0,
                 num_episodes=500, exps=500, epsilon=0.1):
        self.grid = grid
        self.Q = self._init_q()
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.exps = exps
        self.epsilon = epsilon
        self.gamma = gamma
        self.l = l

    def _init_q(self):
        return np.zeros((len(self.grid.P), self.grid.nA))

    def _init_start_state(self):
        return (self.grid.shape[0] - 1) * self.grid.shape[1]

    def _next_action(self, state):
        """ Given a current state, returns the next possible action
            using eps-greedy strategy.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.grid.nA))
        else:
            return np.argmax(self.Q[state])

    def _policy_directions(self, policy):
        translated = [""] * len(policy)
        for i, val in enumerate(policy):
            if val == 0:
                translated[i] = "U"
            elif val == 1:
                translated[i] = "R"
            elif val == 2:
                translated[i] = "D"
            else:
                translated[i] = "L"
        return np.array(translated).reshape(self.grid.shape)

    def _choose_policy(self):
        """ Choose a policy from Q table """
        return np.argmax(self.Q, axis=1)


class QLearner(RL):

    def learn(self, start_state=None):
        time_steps_per_episode, max_q_value_per_episode = np.zeros(
            (1, self.num_episodes)), np.zeros((1, self.num_episodes))
        if not start_state:
            start_state = self._init_start_state()
        for e in xrange(self.exps):
            self.Q = self._init_q()
            for i in xrange(self.num_episodes):
                rewards, state, is_terminal = 0, start_state, False
                while not is_terminal:
                    action = self._next_action(state)
                    new_state, reward, is_terminal = self.grid.move(
                        state, action)
                    self.Q[state, action] += self.alpha * (reward + (self.gamma * (np.max(self.Q[new_state]))) - self.Q[state, action])
                    state, rewards = new_state, rewards + reward
                    time_steps_per_episode[0, i] += 1
                    if is_terminal:
                        break
                max_q_value_per_episode[0, i] += np.max(self.Q[start_state])
                sys.stdout.write("Experiment: {}\t Episodes: {}\r".format(
                    e + 1, i + 1))
                sys.stdout.flush()
        avg_time_steps = time_steps_per_episode / self.exps
        avg_max_q = max_q_value_per_episode / self.exps
        return self._policy_directions(self._choose_policy()), avg_time_steps, avg_max_q


class SarsaLambdaLearner(RL):

    def __init__(self, g, **kwargs):
        super(SarsaLambdaLearner, self).__init__(g, **kwargs)
        self._init_e = self._init_q
        self.e = self._init_e()

    def learn(self, start_state=None):
        time_steps_per_episode, max_q_value_per_episode = np.zeros(
            (1, self.num_episodes)), np.zeros((1, self.num_episodes))
        if not start_state:
            start_state = self._init_start_state()
        for e in xrange(self.exps):
            self.Q = self._init_q()
            for i in xrange(self.num_episodes):
                self.e = self._init_e()
                rewards, state, is_terminal = 0, start_state, False
                action = self._next_action(state)
                while not is_terminal:
                    new_state, reward, is_terminal = self.grid.move(
                        state, action)
                    new_action = self._next_action(state)
                    delta = reward + self.gamma * \
                        self.Q[new_state, new_action] - self.Q[state, action]
                    self.e[state, action] += 1
                    self.Q += self.alpha * delta * self.e
                    self.e = self.gamma * self.l * self.e
                    state, action, rewards = new_state, new_action, rewards + reward
                    time_steps_per_episode[0, i] += 1
                    if is_terminal:
                        break
                max_q_value_per_episode[0, i] += np.max(self.Q[start_state])
                sys.stdout.write("Experiment: {} Total Rewards after {} experiments = {} \r".format(
                    e + 1, i + 1, rewards))
                sys.stdout.flush()
        avg_time_steps = time_steps_per_episode / self.exps
        avg_max_q = max_q_value_per_episode / self.exps
        return self._policy_directions(self._choose_policy()), avg_time_steps, avg_max_q


def plot(v):
    import matplotlib.pylab as plt
    fig, ax = plt.subplots()
    min_val, max_val = 0, 5
    for i in xrange(5):
        for j in xrange(5):
            c = v[i][j]
            ax.text(i, j, str(c), va='center', ha='center')
    ax.matshow(v, cmap=plt.cm.Blues)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks(np.arange(max_val))
    ax.set_yticks(np.arange(max_val))
    ax.grid()
    plt.show()


if __name__ == '__main__':
    shape = (5, 5)
    g = GridworldEnv(shape=shape)
    l = SarsaLambdaLearner(g, exps=2, l=0.8, num_episodes=400,
                           gamma=0.99, alpha=0.1, epsilon=0.3)
    print l.learn()
