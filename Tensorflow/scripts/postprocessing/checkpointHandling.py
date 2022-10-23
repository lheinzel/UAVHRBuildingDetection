import time
from shutil import rmtree
import os

def moveCheckpointFilesToFolder(modelPath, destPath, lastCheckpoint, maxRunTime):
    startTime = time.time()
    
    # Clear existing checkpoints if present
    if os.path.exists(destPath):
        rmtree(destPath)
    os.mkdir(destPath)

    elapsedTime = time.time()

    while elapsedTime-startTime < maxRunTime:
        # Read names of all checkpoint related files
        dirContents = os.scandir(modelPath)
        ckptFilePaths = [el.path for el in dirContents if el.is_file() and el.name.split("-")[0] == "ckpt"] 
        
        # Get number of latest checkpoint
        ckptNumbers = [int(os.path.split(el)[1].split(".")[0].split("-")[1]) for el in ckptFilePaths]
        latestCkptNum = max(ckptNumbers)

        # Move over all checkpoint related files
        for ckptFile in ckptFilePaths:
            ckptFileName = os.path.split(ckptFile)[1]
            os.rename(ckptFile,os.path.join(destPath, ckptFileName))

        # Break if the last checkpoint has been moved
        if latestCkptNum >= lastCheckpoint:
            break;

        # Update elabsed time
        elapsedTime = time.time();

if __name__ == "__main__":
    modelPath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2";
    destPath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2/checkpoints"
    lastCheckpoint = 21
    maxRunTime = 120

    moveCheckpointFilesToFolder(modelPath, destPath, lastCheckpoint, maxRunTime)
        
    