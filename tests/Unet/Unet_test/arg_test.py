from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
args = parser.parse_args()

print(args.model_config)