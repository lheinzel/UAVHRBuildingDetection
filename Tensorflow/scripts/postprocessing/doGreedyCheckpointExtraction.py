from checkpointHandling import copyCheckpointFilesToFolder
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description="Script for greedy checkpoint extraction whilst training")
parser.add_argument("-m","--modelDir",help="directory of the model",type=str)
parser.add_argument("-d","--destDir", help="destination of the checkpoint files", type=str)
parser.add_argument("-l","--lastCkpt", help="number of the last checkpoint", type=int)
parser.add_argument("-r", "--maxRunTime", help="maximum runtime (minutes) of the script", type=int)

args = parser.parse_args()

def main():
    copyCheckpointFilesToFolder(args.modelDir, args.destDir, args.lastCkpt, args.maxRunTime)

if __name__ == "__main__":
    main()
