import os
import shutil as sh
import argparse
import random as rnd


def iterateImageDir(dirPath, destPath, imgFileExt, lFileNamesTrain, lFileNamesTest, lFileNamesValid, ratioTest, ratioValid):
    destPathTest = os.path.join(destPath, "test")
    destPathTrain = os.path.join(destPath, "train")
    destPathValid = os.path.join(destPath, "valid")

    # Get all elements in current dir
    dirElements = os.listdir(dirPath)

    # partition images in current directory
    partitionCurrentImageDirectory(dirPath, imgFileExt, destPathTest, destPathTrain, destPathValid, lFileNamesTrain, 
                                   lFileNamesTest, lFileNamesValid, ratioTest, ratioValid)

    # Recursive function call for subdirectories
    for el in os.scandir(dirPath):
        if el.is_dir():
            iterateImageDir(el.path, destPath, imgFileExt, lFileNamesTrain, lFileNamesTest, lFileNamesValid, ratioTest, ratioValid)


def partitionCurrentImageDirectory(dirPath, imgFileExt, destTest, destTrain, destValid, lFileNamesTrain, lFileNamesTest, lFileNamesValid,
                                   ratioTest, ratioValid):
    # Get all image files in the current directory
    dirContent = os.listdir(dirPath)
    imgFilesDir = [e for e in dirContent if not os.path.isdir(os.path.join(dirPath,e)) and e.split(".")[1] == imgFileExt]
  
    # Calculate number of images to be copied, in the test folder
    nImagesTest = round(ratioTest*len(imgFilesDir))
    nImagesvalidation = round(ratioValid*len(imgFilesDir)) 
    nImagesTrain = len(imgFilesDir) - nImagesTest - nImagesvalidation
    

    # Copy nImagesTest random images to the test folder
    for i in range(nImagesTest):
        # Select file ranomly and delete it from the list to achieve unique selections
        imgCurrent = rnd.choice(imgFilesDir)
        imgFilesDir.remove(imgCurrent)

        # Copy the file
        sh.copyfile(os.path.join(dirPath, imgCurrent), os.path.join(destTest, imgCurrent))

        # Add file name to list of copied files
        lFileNamesTest.append(imgCurrent.split(".")[0])

    # Perform same again for validation
    for i in range(nImagesvalidation):
        imgCurrent = rnd.choice(imgFilesDir)
        imgFilesDir.remove(imgCurrent)

        sh.copyfile(os.path.join(dirPath, imgCurrent), os.path.join(destValid, imgCurrent))

        lFileNamesValid.append(imgCurrent.split(".")[0])

    # copy the remaining files into the train folder
    for f in imgFilesDir:
        sh.copyfile(os.path.join(dirPath, f), os.path.join(destTrain, f))
        lFileNamesTrain.append(f.split(".")[0])

    
def copyLabelsForImages(lblDirPath, destTest, destTrain, destValid, lFileNamesTrain, lFileNamesTest, lFileNmamesValid, imgFileExt):
    dirContents = os.scandir(lblDirPath)

    # Iterate over directory content
    for el in dirContents:
        # if the file is a .csv file: copy it to the respective destination if it matches an image name
        # of the test or train partition
        if el.is_file() and el.name.split(".")[1] == "csv":

            if el.name.split(".")[0] in lFileNamesTrain:
                sh.copyfile(el.path, os.path.join(destTrain,el.name))
                lFileNamesTrain.remove(el.name.split(".")[0])

            elif el.name.split(".")[0] in lFileNmamesValid:
                sh.copyfile(el.path, os.path.join(destValid, el.name))
                lFileNmamesValid.remove(el.name.split(".")[0])

            elif el.name.split(".")[0] in lFileNamesTest:
                sh.copyfile(el.path, os.path.join(destTest, el.name))
                lFileNamesTest.remove(el.name.split(".")[0])

        # if it is a directory, perform recursion
        else:
            copyLabelsForImages(el.path, destTest, destTrain, destValid, lFileNamesTrain, lFileNamesTest, lFileNmamesValid, imgFileExt)



def partition(imgSource, labelSource, dest, ratioTest, ratioValid, imgFileExt):
    destTest = os.path.join(dest, "test")
    destTrain = os.path.join(dest, "train")
    destValid = os.path.join(dest, "valid")

    if not os.path.exists(destTrain):
        os.makedirs(destTest)

    if not os.path.exists(destTrain):
        os.makedirs(destTrain)

    if not os.path.exists(destValid):
        os.makedirs(destValid)
    

    # initialize lists of image files that have been copied
    lFileNamesTest = [];
    lFileNamesTrain = [];
    lFileNamesValid = []

    # Copy all image files according to partitioning into the target diectories, by iterating over the source directores. 
    # Save the names in the lists
    print("Partitioning the images...")
    iterateImageDir(imgSource, dest, imgFileExt, lFileNamesTrain, lFileNamesTest, lFileNamesValid, ratioTest, ratioValid)

    # Copy the matching labels to the respective target directories of the partitions
    print("Copying matching labels...")
    copyLabelsForImages(labelSource, destTest, destTrain, destValid, lFileNamesTrain, lFileNamesTest, lFileNamesValid, imgFileExt)

    # Verify the partitioning
    print("Verifying Test partition...")
    diffTest = verifyPartitioning(destTest, imgFileExt)#
    if diffTest:
        print("Diff:")
        print(list(diffTest))
        raise ValueError("Test partitioning not successful!")
    else:
        print("Test partition o.k. !")

    print("Verifying Train partition...")
    diffTrain = verifyPartitioning(destTrain, imgFileExt)
    if diffTrain:
        print("Diff:")
        print(list(diffTrain))
        raise ValueError("Train partitioning not successful!")
    else:
        print("Train partition o.k. !")

    if ratioValid > 0:
        print("Verifying Validation partition...")
        diffValid = verifyPartitioning(destValid, imgFileExt)
        if diffValid:
            print("Diff:")
            print(list(diffValid))
            raise ValueError("Validation partitioning not successful!")
        else:
            print("Validation partition o.k. !")



def verifyPartitioning(dest, imgFileExt):
    # Create sets of image and label files
    dirContent = os.listdir(dest)
    imgFiles = set([f.split(".")[0] for f in dirContent if f.split(".")[1] == imgFileExt])
    labelFiles = set([f.split(".")[0] for f in dirContent if f.split(".")[1] == "csv"])

    # Compute difference of the sets
    contentDiff = imgFiles.difference(labelFiles)

    return contentDiff


if __name__ == '__main__':
    imgSourcePath = r"Datasets/UAVHighRiseBuildingsKorea/DataCropped_320x320/Images"
    lblSourcePath = r"Datasets/UAVHighRiseBuildingsKorea/DataCropped_320x320/Labels"
    destPath = r"Tensorflow/workspace/images"
    ratio = 0.1
    fileNamesTest = []
    fileNamesTrain = []
  
    partition(imgSourcePath, lblSourcePath, destPath, ratio, ratio, "png")
  
