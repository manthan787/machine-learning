import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv():
    """
    You are an agent on an s x s grid and your goal is to reach the terminal
    state at the top right corner.
    For example, a 4x4 grid looks as follows:
    o  o  o  T
    o  o  o  o
    o  o  o  o
    x  o  o  o

    x is your position and T is the terminal state.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -0.1 at each step until you reach a terminal state.
    """

    def __init__(self, shape=[4, 4]):
        self.shape = shape

        nS = np.prod(shape)  # The area of the gridworld
        MAX_Y = shape[0]
        MAX_X = shape[1]
        self.nA = 4  # There are four possible actions
        self.P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex    # s is the current position id. s = y * 4 + x           
            y, x = it.multi_index

            self.P[s] = {a: [] for a in range(self.nA)}

            def is_done(s): return s == shape[1] - 1
            reward = 5.0 if is_done(s) else -0.1

            # We're stuck in a terminal state
            if is_done(s):
                self.P[s][UP] = [(s, reward, True)]
                self.P[s][RIGHT] = [(s, reward, True)]
                self.P[s][DOWN] = [(s, reward, True)]
                self.P[s][LEFT] = [(s, reward, True)]            
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                self.P[s][UP] = [(ns_up, reward, is_done(ns_up))]
                self.P[s][RIGHT] = [(ns_right, reward, is_done(ns_right))]
                self.P[s][DOWN] = [(ns_down, reward, is_done(ns_down))]
                self.P[s][LEFT] = [(ns_left, reward, is_done(ns_left))]
            # print y, x, s
            it.iternext()

    # The possible action has a 0.8 probability of succeeding
    def action_success(self, success_rate=0.8):
        return np.random.choice(2, 1, p=[1-success_rate, success_rate])[0]

    # If the action fails, any action is chosen uniformly(including the succeeding action)
    def get_action(self, action):
        if self.action_success():
            return action
        else:
            random_action = np.random.choice(
                4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
            return random_action

    # Given the current position, this function outputs the position after the action.
    def move(self, s, action):
        return self.P[s][self.get_action(action)][0]


if __name__ == "__main__":
    g = GridworldEnv((5, 5))
    print(g.move(12, g.get_action(0)))
    print(g.P.keys())
