from rl import QLearner, SarsaLambdaLearner
from gridworld import GridworldEnv
import matplotlib.pyplot as plt
import numpy as np


class Config(object):

    def __init__(self, name, learner, size=5, **params):
        self.name = name
        self.grid = GridworldEnv((size, size))        
        if learner == 'q':
            self.learner = QLearner(self.grid, **params)
        if learner == 's':
            self.learner = SarsaLambdaLearner(self.grid, **params)


class Analyzer(object):
    """ Perform analysis with given configurations """

    def __init__(self, analysis_name):
        self.analysis_name = analysis_name
        self.max_q_values = {}
        self.avg_time_steps = {}
        self.configs = []

    def add_config(self, config):
        self.configs.append(config)

    def run(self):
        for config in self.configs:
            print "Running Config: {}".format(config.name)
            policy, avg_time_steps, max_q_value = config.learner.learn()
            self.max_q_values[config] = max_q_value
            self.avg_time_steps[config] = avg_time_steps
        if isinstance(config.learner, SarsaLambdaLearner):
            title = "Max Q value of start state per episode (lambda={})".format(
                config.learner.l)
        else:
            title = "Max Q value of start state per episode"
        self.plot_(self.max_q_values, "maxQ", title,
                   "Max Q Value for Start State")
        if isinstance(config.learner, SarsaLambdaLearner):
            title = "Average number of time steps per episode (lambda={})".format(
                config.learner.l)
        else:
            title = "Average number of time steps per episode"
        self.plot_(self.avg_time_steps, "ts", title,
                   "Average number of timesteps")

    def plot_(self, values, value_type, title, ylab):
        for config, val in values.iteritems():
            plt.plot(range(config.learner.num_episodes),
                     val[0], label="Alpha = {}, Eps = {}, lambda={}".format(config.learner.alpha, config.learner.epsilon, config.learner.l))
        plt.xlabel("Number of Episodes")
        plt.ylabel(ylab)
        plt.title(title)
        plt.legend()
        plt.savefig("plots/{}-{}.png".format(self.analysis_name, value_type),
                    bbox_inches='tight')
        plt.clf()


def analyse_alpha(learner, size=5, exps=500, num_episodes=500):
    alphas = [0, 0.01, 0.05, 0.1, 0.5, 1]
    a = Analyzer("analysis-alpha")
    for alpha in alphas:
        config_name = "alpha-{}-learner-{}-size-{}".format(
            alpha, learner, size)
        params = {'alpha': alpha, 'num_episodes': num_episodes, 'exps': exps}
        a.add_config(Config(config_name, learner, size=size, **params))
    a.run()


def analyse_epsilon(learner, alpha=0, size=5, exps=500, num_episodes=500):
    epsilons = [0, 0.1, 0.25, 0.5, 1]
    a = Analyzer("alpha-{}-epsilon-{}".format(alpha, learner))
    for epsilon in epsilons:
        config_name = "alpha-{}-eps-{}-learner-{}-size-{}".format(
            alpha, epsilon, learner, size)
        params = {'alpha': alpha, 'epsilon': epsilon,
                  'num_episodes': num_episodes, 'exps': exps}
        a.add_config(Config(config_name, learner, size=size, **params))
    a.run()


def analyse_lambda(learner, alpha=0, l=0, size=5, exps=500, num_episodes=500):
    # alphas, epsilons = [0, 0.01, 0.05, 0.1, 0.5, 1], [0, 0.1, 0.25, 0.5, 1]
    alpha, epsilon = 0.5, 0.25
    lambdas_ = [0, 0.25, 0.5, 0.75, 1]
    a = Analyzer("alpha-{}-epsilon-{}-lambdas".format(alpha, epsilon))
    for l in lambdas_:        
        config_name = "alpha-eps-lambdas"
        params = {'alpha': alpha, 'epsilon': epsilon, 'l': l,
                    'num_episodes': num_episodes, 'exps': exps}
        a.add_config(Config(config_name, learner, size=size, **params))
    a.run()


if __name__ == '__main__':
    # analyse_lambda('s', exps=50)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-alg", required=True,
                        help="""Algorithm to be performed.
                        Must be either 'q' (QLearning) or 's' (SarsaLambda)""")
    parser.add_argument("-size", help="Size of the grid world")
    parser.add_argument("-alpha", help="Learning Rate")
    parser.add_argument("-exps", help="No of experiments to perform")
    parser.add_argument(
        "-eps", help="No of episodes to perform within each experiment")
    parser.add_argument("-l", help="Value of lambda")
    args = parser.parse_args()
    config = {}
    if args.eps:
        config['num_episodes'] = int(args.eps)
    if args.exps:
        config['exps'] = int(args.exps)
    if not args.alpha and not args.l:
        analyse_alpha(args.alg, **config)
    elif not args.l:
        analyse_epsilon(args.alg, alpha=float(args.alpha), **config)
    else:
        analyse_lambda('s', l=float(args.l), **config)
