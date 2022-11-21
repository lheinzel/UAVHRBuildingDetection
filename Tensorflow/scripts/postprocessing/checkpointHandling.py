import time
from shutil import rmtree
import os
from shutil import copyfile
import tarfile

def copyCheckpointFilesToFolder(modelPath, destPath, lastCheckpoint, maxRunTime):
    print("... Start copyCheckpointFilesToFolder")
    startTime = time.time()
    
    # Clear existing checkpoints if present
    if os.path.exists(destPath):
        rmtree(destPath)
    os.mkdir(destPath)

    elapsedTime = time.time()

    while (elapsedTime-startTime)/60 < maxRunTime:
        # Read names of all checkpoint related files
        dirContents = os.scandir(modelPath)

        ckptFilePaths = [el.path for el in dirContents if el.is_file() and el.name.split("-")[0] == "ckpt"] 

        # make sure that only fully built files are moved
        if len(ckptFilePaths)>=4:
            # Get number of latest checkpoint
            ckptNumbers = [int(os.path.split(el)[1].split(".")[0].split("-")[1]) for el in ckptFilePaths]
            latestCkptNum = max(ckptNumbers)

            # Move over the files of the oldest checkpoint
            ckptNumbers = list(set(ckptNumbers))
            ckptNumbers.sort()

            for f in ckptFilePaths:
                if str(ckptNumbers[0]) in os.path.split(f)[1].split(".")[0]:
                    fileName = os.path.split(f)[1]
                    print("moving file " + fileName)
                    os.rename(f, os.path.join(destPath, fileName))
           

            # Break if the last checkpoint is present
            if latestCkptNum >= lastCheckpoint:
                break;

        # Update elabsed time
        elapsedTime = time.time();

    # Move remaining checkpoint related files
    dirContents = os.scandir(modelPath)
    for el in dirContents:
        if el.is_file() and ("ckpt" in el.name or "checkpoint" in el.name):
            print("moving file " + el.name)
            os.rename(el.path, os.path.join(destPath, el.name))

    print("Latest checkpoint copied. Quitting ...")
    

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

    # Copy the base checkpoint index file
    if "checkpoint" in ckptFiles:
        copyfile(os.path.join(ckptSourcePath,"checkpoint"),os.path.join(ckptTargetPath, "checkpoint"))

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
    ckptFileNums = [int(el.split(".")[0].split("-")[1]) for el in ckptFiles if "ckpt-" in el and not ".log" in el]
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


def setCheckpointPointerInteratively(checkpointPath, evalPath, maxRunTime):
    startTime = time.time()
    ckptPointerFilePath = os.path.join(checkpointPath, "checkpoint")

    # remove existing pointer file if exists
    if os.path.exists(ckptPointerFilePath):
        print("... delete old pointer file")
        os.remove(ckptPointerFilePath)

    # Get all checkpoint files and the indizes
    ckptFiles = os.listdir(checkpointPath)
    ckptIndices = getCheckpointIndices(checkpointPath)
    ckptIndices.sort()

    # Set initial pointer to first checkpoint
    setCheckpointPointer(ckptPointerFilePath, ckptIndices.pop(0))

    # read elapsed time and number of created evaluation files
    elapsedTime = time.time()
    numEvalFilesPrev = len(os.listdir(evalPath))

    while (elapsedTime-startTime)/60 < maxRunTime and ckptIndices:
        numEvalFilesCur = len(os.listdir(evalPath))

        # Set pointer to next checkpoint if previous one has been read
        if numEvalFilesCur >= numEvalFilesPrev + 1:
            setCheckpointPointer(ckptPointerFilePath, ckptIndices.pop(0))
            numEvalFilesPrev = numEvalFilesCur;

        elapsedTime = time.time()

def setCheckpointPointer(ckptPointerFilePath, index):
    print("... Setting pointer to " + "\"ckpt-" + str(index) + "\"")
    pointerFile = open(ckptPointerFilePath, "w")
    pointerFile.write("model_checkpoint_path: \"ckpt-" + str(index) + "\"")
    pointerFile.close()



if __name__ == "__main__":
    modelPath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2";
    destPath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2/checkpoints"
    evalpath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2/eval"
    ckptBufferTarget = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2/ckptBuffer"
    ckptTartPath = r"Tensorflow/workspace/training_SSD-MobnetV2_320x320_MoreAugments/models/HRDetection_MobNetV2/checkpoints.tar.gz"
    lastCheckpoint = 21
    maxRunTime = 120

    #copyCheckpointFilesToFolder(modelPath, destPath, lastCheckpoint, maxRunTime)
    #copyCheckpontFilesForEvaluation(destPath, ckptBufferTarget, evalpath, 30)
    #saveCheckpointDataToCloud(destPath, ckptTartPath)

    setCheckpointPointerInteratively(destPath, evalpath, maxRunTime)