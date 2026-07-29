import argparse
from train import train
from generate import generate

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices = ["train", "generate"])

args = parser.parse_args()

if args.mode == "train":
    train()
elif args.mode == "generate":
    generate()
else:
    print("error: the following arguments are required: --mode (use 'train' or 'generate')")
