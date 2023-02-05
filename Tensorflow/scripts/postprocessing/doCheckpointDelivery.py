from checkpointHandling import copyCheckpontFilesForEvaluation
import argparse

parser = argparse.ArgumentParser(description="script for continuous delivery of checkpoint files for evaluation")
parser.add_argument("-s","--ckptSrc",help="path to checkpoint source directory",type=str)
parser.add_argument("-t","--ckptTar",help="target path to deliver checkpoint files to", type=str)
parser.add_argument("-e","--evalPath",help="directory in which the eval files are stored", type=str)
parser.add_argument("-m","--maxRuntime",help="maximum runtime of the script", type=int)
args = parser.parse_args()


def main():
    copyCheckpontFilesForEvaluation(args.ckptSrc, args.ckptTar, args.evalPath, args.maxRuntime);

if __name__=="__main__":
    main()