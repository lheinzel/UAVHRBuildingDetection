import os
import shutil as sh
import argparse
import random as rnd


def iterateImageDir(dirPath, destPath, imgFileExt, lFileNamesTrain, lFileNamesTest, ratio):
    destPathTest = os.path.join(destPath, "Test")
    destPathTrain = os.path.join(destPath, "Train")

    # Get all elements in current dir
    dirElements = os.listdir(dirPath)

    # partition images in current directory
    partitionCurrentImageDirectory(dirPath, imgFileExt, destPathTest, destPathTrain, lFileNamesTrain, lFileNamesTest, ratio)

    # Recursive function call for subdirectories
    for el in os.scandir(dirPath):
        if el.is_dir():
            iterateImageDir(el.path, destPath, imgFileExt, lFileNamesTrain, lFileNamesTest, ratio)


def partitionCurrentImageDirectory(dirPath, imgFileExt, destTest, destTrain, lFileNamesTrain, lFileNamesTest, ratio):
    # Get all image files in the current directory
    dirContent = os.listdir(dirPath)
    imgFilesDir = [e for e in dirContent if not os.path.isdir(os.path.join(dirPath,e)) and e.split(".")[1] == imgFileExt]
  
    # Calculate number of images to be copied, in the test folder
    nImagesTest = round(ratio*len(imgFilesDir))
    nImagesTrain = len(imgFilesDir) - nImagesTest

    # Copy nImagesTest random images to the test folder
    for i in range(nImagesTest):
        # Select file ranomly and delete it from the list to achieve unique selections
        imgCurrent = rnd.choice(imgFilesDir)
        imgFilesDir.remove(imgCurrent)

        # Copy the file
        sh.copyfile(os.path.join(dirPath, imgCurrent), os.path.join(destTest, imgCurrent))

        # Add file name to list of copied files
        lFileNamesTest.append(imgCurrent.split(".")[0])

    # copy the remaining files into the train folder
    for f in imgFilesDir:
        sh.copyfile(os.path.join(dirPath, f), os.path.join(destTrain, f))
        lFileNamesTrain.append(f.split(".")[0])

    
def copyLabelsForImages(lblDirPath, destTest, destTrain, lFileNamesTrain, lFileNamesTest, imgFileExt):
    dirContents = os.scandir(lblDirPath)

    # Iterate over directory content
    for el in dirContents:
        # if the file is a .csv file: copy it to the respective destination if it matches an image name
        # of the test or train partition
        if el.is_file() and el.name.split(".")[1] == "csv":

            if el.name.split(".")[0] in lFileNamesTrain:
                sh.copyfile(el.path, os.path.join(destTrain,el.name))
                lFileNamesTrain.remove(el.name.split(".")[0])

            elif el.name.split(".")[0] in lFileNamesTest:
                sh.copyfile(el.path, os.path.join(destTest, el.name))
                lFileNamesTest.remove(el.name.split(".")[0])

        # if it is a directory, perform recursion
        else:
            copyLabelsForImages(el.path, destTest, destTrain, lFileNamesTrain, lFileNamesTest, imgFileExt)



def partition(imgSource, labelSource, dest, ratio, imgFileExt):
    destTest = os.path.join(dest, "test")
    destTrain = os.path.join(dest, "train")

    # initialize lists of image files that have been copied
    lFileNamesTest = [];
    lFileNamesTrain = [];

    # Copy all image files according to partitioning into the target diectories, by iterating over the source directores. 
    # Save the names in the lists
    print("Partitioning the labels...")
    iterateImageDir(imgSource, dest, imgFileExt, lFileNamesTrain, lFileNamesTest, ratio)

    # Copy the matching labels to the respective target directories of the partitions
    print("Copying matching labels...")
    copyLabelsForImages(labelSource, destTest, destTrain, lFileNamesTrain, lFileNamesTest, imgFileExt)

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




def verifyPartitioning(dest, imgFileExt):
    # Create sets of image and label files
    dirContent = os.listdir(dest)
    imgFiles = set([f.split(".")[0] for f in dirContent if f.split(".")[1] == imgFileExt])
    labelFiles = set([f.split(".")[0] for f in dirContent if f.split(".")[1] == "csv"])

    # Compute difference of the sets
    contentDiff = imgFiles.difference(labelFiles)

    return contentDiff


if __name__ == '__main__':
    imgSourcePath = r"Datasets/UAVHighRiseBuildingsKorea/DataAugmented/Images"
    lblSourcePath = r"Datasets/UAVHighRiseBuildingsKorea/DataAugmented/Labels"
    destTest = r"Tensorflow/workspace/images/Test"
    destTrain = r"Tensorflow/workspace/images/Train"
    destPath = r"Tensorflow/workspace/images"
    ratio = 0.1
    fileNamesTest = []
    fileNamesTrain = []
  
    partition(imgSourcePath, lblSourcePath, destPath, ratio, "png")
  
