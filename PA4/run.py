import argparse
from rl import QLearner, SarsaLambdaLearner
from gridworld import GridworldEnv

DEFAULT_GRID_SIZE = 5


def args_handler(parser):
    args = parser.parse_args()
    config, size = {}, DEFAULT_GRID_SIZE
    if args.size:
        size = int(args.size)
    grid = GridworldEnv((size, size))
    if args.gamma:
        config['gamma'] = float(args.gamma)
    if args.exps:
        config['exps'] = int(args.exps)
    if args.eps:
        config['num_episodes'] = int(args.eps)
    if args.epsilon:
        config['epsilon'] = float(args.epsilon)
    if args.alpha:
        config['alpha'] = float(args.alpha)
    if args.lamb:
        config['l'] = float(args.lamb)
    if args.alg not in ['q', 's']:
        parser.print_help()
        return
    elif args.alg == 'q':
        learner = QLearner(grid, **config)
    elif args.alg == 's':
        learner = SarsaLambdaLearner(grid, **config)    
    print "\n"
    print  learner.learn()[0]
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-alg", required=True,
                        help="""Algorithm to be performed.
                        Must be either 'q' (QLearning) or 's' (SarsaLambda)""")
    parser.add_argument("-size", help="Size of the gridworld")
    parser.add_argument("-gamma", help="Return Decay parameter")
    parser.add_argument("-exps", help="No of experiments to perform")
    parser.add_argument(
        "-eps", help="No of episodes to perform within each experiment")
    parser.add_argument(
        "-epsilon", help="Value of epsilon for e-greedy policy")
    parser.add_argument("-alpha", help="Learning rate")
    parser.add_argument(
        "-lamb", help="Eligibiility trace decay parameter")
    args_handler(parser)
