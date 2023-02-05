from checkpointHandling import setCheckpointPointerInteratively
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description="Script for iteratively setting the pointer to the next checkpoint for evaluation of the model")
parser.add_argument("-c","--ckptPath",help="Path to checkpoint files",type=str)
parser.add_argument("-e","--evalPath",help="Path to directory of evaluation files",type=str)
parser.add_argument("-m","--maxRuntime",help="max runtime of the script",type=int)

args = parser.parse_args()

def main():
    setCheckpointPointerInteratively(args.ckptPath, args.evalPath, args.maxRuntime)


if __name__ == "__main__":
    main()
