import argparse


parser = argparse.ArgumentParser(description='Parse the configurations for the experiments')

parser.add_argument('-x','--experiment', dest='exp',
                    help='Name of the experiment')

parser.add_argument('-d', '--dataset', dest='dataset',
                    choices=['mnist', 'fmnist'], default='mnist',
                    help='Data set for training and evaluating')

parser.add_argument('--epochs', dest='epochs', type=int, default='70',
                    help='Number of training epochs')

parser.add_argument('--m_dir', dest='model_dir',
                    default='trained_models/',
                    help='Directory where models are saved')

parser.add_argument('--random_seed', dest='random_seed', type=int,
                    default=845234, help='Random seed')

parser.add_argument('--num_summary', dest='num_summary_steps', type=int,
                    default=200, help='Number of summary steps for logging results')

parser.add_argument('--num_checkpoint', dest='num_checkpoint_steps', type=int,
                    default=500, help='Number of checkpoint steps for saving models')

parser.add_argument('--num_eval', dest='num_eval_steps', type=int,
                    default=1000, help='Number of steps between evaluations')

parser.add_argument('-b', '--train_batch', dest='batch_size', type=int,
                    default=32, help='Training batch size')

parser.add_argument('--l_rate', dest='l_rate', type=float, default=1e-4,
                    help='Learnig rate')

parser.add_argument('--eval_batch', dest='eval_batch_size', type=int,
                    default=200, help='Evaluation batch size')

parser.add_argument('-e', dest='eval_during_training', action='store_true',
                    help='Evaluation during training')

args = parser.parse_args()
