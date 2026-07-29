import argparse
from train import train
from generate import generate

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices = ["train", "generate"])
parser.add_argument("--resume", action = "store_true")

args = parser.parse_args()

if args.mode == "train":
    train(args.resume)

elif args.mode == "generate":
    out_file = open("outputs/output.txt", "w")
    out_file.write(generate())

else:
    print("error: the following arguments are required: --mode (use 'train' or 'generate')")
