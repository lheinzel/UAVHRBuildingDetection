from multiprocessing.sharedctypes import Value
import time
from shutil import rmtree
import os
from shutil import copyfile
import tarfile

def moveCheckpointFilesToFolder(modelPath, destPath, lastCheckpoint, maxRunTime):
    startTime = time.time()
    
    # Clear existing checkpoints if present
    if os.path.exists(destPath):
        rmtree(destPath)
    os.mkdir(destPath)

    elapsedTime = time.time()

    while (elapsedTime-startTime < maxRunTime)/60:
        # Read names of all checkpoint related files
        dirContents = os.scandir(modelPath)

        ckptFilePaths = [el.path for el in dirContents if el.is_file() and el.name.split("-")[0] == "ckpt"] 

        if ckptFilePaths:
            # Get number of latest checkpoint
            ckptNumbers = [int(os.path.split(el)[1].split(".")[0].split("-")[1]) for el in ckptFilePaths]
            latestCkptNum = max(ckptNumbers)

            # Move over all checkpoint related files
            for ckptFile in ckptFilePaths:
                ckptFileName = os.path.split(ckptFile)[1]
                print("Moving file " + ckptFile)
                os.rename(ckptFile,os.path.join(destPath, ckptFileName))

            # Break if the last checkpoint has been moved
            if latestCkptNum >= lastCheckpoint:
                print("Latest checkpoint moved. Quitting ...")
                break;

        # Update elabsed time
        elapsedTime = time.time();

def copyCheckpontFilesForEvaluation(ckptSourcePath, ckptTargetPath, evalPath, maxRunTime):
    startTime = time.time()

    # Clear existing checkpoint buffer if present
    if os.path.exists(ckptTargetPath):
        rmtree(ckptTargetPath)
    os.mkdir(ckptTargetPath)
    
    # Get all checkpoint files and the indizes
    ckptFiles = os.listdir(ckptSourcePath)
    ckptIndices = getCheckpointIndices(ckptSourcePath)
    ckptIndices.sort()

    # Copy files of first checkpoint
    copyCheckpointFilesForIndex(ckptFiles, ckptSourcePath, ckptIndices.pop(0), ckptTargetPath)

    elapsedTime = time.time()
    numEvalFilesPrev = len(os.listdir(evalPath))

    # Copy the remaining checkpoints one after another after the previous has been processed
    while (elapsedTime-startTime)/60 < maxRunTime and ckptIndices:
        numEvalFilesCur = len(os.listdir(evalPath))

        # Copy next checkpoint file if previous has been processed
        if numEvalFilesCur > numEvalFilesPrev:
            copyCheckpointFilesForIndex(ckptFiles, ckptSourcePath, ckptIndices.pop(0), ckptTargetPath)
            numEvalFilesPrev = numEvalFilesCur

        elapsedTime = time.time()


def getCheckpointIndices(ckptSourcePath):
    ckptFiles = os.listdir(ckptSourcePath)
    ckptFileNums = [int(el.split(".")[0].split("-")[1]) for el in ckptFiles]
    ckptIndizes = list(set(ckptFileNums))
    return ckptIndizes

def copyCheckpointFilesForIndex(ckptFileNames, ckptSrcPath, index, ckptTargetPath):
    # Get the names of the requiered file for the checkpoint
    ckptFilesCurrent = [el for el in ckptFileNames if str(index) in el.split(".")[0]]

    # Copy all the files
    print("...Copying files for Checkpoint: " + str(index))
    for f in ckptFilesCurrent:
        copyfile(os.path.join(ckptSrcPath,f), os.path.join(ckptTargetPath, f))

def saveCheckpointDataToCloud(ckptPath, ckptTarPath):
    if os.path.exists(ckptTarPath):
        raise ValueError("Checkpoint data at " + ckptTarPath + " already exists!")
    else:
        with tarfile.open(ckptTarPath, "w:gz") as tar:
            tar.add(ckptPath, arcname=os.path.basename(ckptPath))

if __name__ == "__main__":
    modelPath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2";
    destPath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2/checkpoints"
    evalpath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2/eval"
    ckptBufferTarget = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2/ckptBuffer"
    ckptTartPath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2/checkpoints.tar.gz"
    lastCheckpoint = 21
    maxRunTime = 120

    moveCheckpointFilesToFolder(modelPath, destPath, lastCheckpoint, maxRunTime)
    #copyCheckpontFilesForEvaluation(destPath, ckptBufferTarget, evalpath, 30)
    #saveCheckpointDataToCloud(destPath, ckptTartPath)