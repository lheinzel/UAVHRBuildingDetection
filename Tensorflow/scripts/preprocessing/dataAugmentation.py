from ctypes.wintypes import tagMSG
from email.mime import image
import enum
import albumentations as alb
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os

def readAnnotationData(lblPath):
    dfLblData = pd.read_csv(lblPath, sep=";")
    listAnnotation = dfLblData[['xmin' , 'ymin', 'xmax', 'ymax', 'class']].values.tolist()
    return listAnnotation


def transformData(oImage, lAnnotation, imgCrpDims, vProb, nAugments):
    dataTransformed = [];

    for i in range(nAugments):
        bboxParams = alb.BboxParams(format='pascal_voc')
        augmenter = alb.Compose([alb.RandomCrop(imgCrpDims[0],imgCrpDims[1]),
                            alb.Affine(shear=[-2.5,2.5], rotate=[-2.5 ,2.5], fit_output=True, p=vProb[0]), 
                            alb.Resize(imgCrpDims[0]+50,imgCrpDims[1]+50),
                            alb.CenterCrop(imgCrpDims[0],imgCrpDims[1]),
                            alb.ColorJitter(hue=0.1, brightness=0.5, saturation=0.5, p=vProb[1]),
                            alb.HorizontalFlip(p=vProb[2]),
                            alb.VerticalFlip(p=vProb[3])], bboxParams)

        dataTransCur = augmenter(image=oImage, bboxes=lAnnotation)
        dataTransformed.append(dataTransCur)

    return dataTransformed


def saveAugmentedData(dataTransformed, targetDir, srcImageName):
    cols = ["filename", "width" ,"height" ,"class", "xmin" ,"ymin" ,"xmax" ,"ymax"]

    # Create datafames for all cropped data
    for ind, dat in enumerate(dataTransformed):
        # Prepare dataframe for labels
        dfAnnotations = pd.DataFrame(columns=cols, index=range(len(dat["bboxes"])))

        imgNameCur = srcImageName + "_" + str(ind) + ".png"
        lblNameCur = srcImageName + "_" + str(ind) + ".csv"

        dfAnnotations["filename"] = imgNameCur
        dfAnnotations["width"] = dat["image"].shape[1]
        dfAnnotations["height"] = dat["image"].shape[0]

        if dat["bboxes"]:
            dfAnnotations[["xmin", "ymin", "xmax", "ymax", "class"]] = dat["bboxes"]
        pass

        # Write Annotation data
        dfAnnotations.to_csv(os.path.join(targetDir, lblNameCur), sep=";")

        # Write image
        cv2.imwrite(os.path.join(targetDir, imgNameCur), dat["image"])




def augmentData(imgDir, lblDir, targetDir, imgCrpDims, nAugments, vProb, imgFileExt):
    for el in os.scandir(imgDir):
        if el.is_file() and el.name.split(".")[1] == imgFileExt:
            lblPath = os.path.join(lblDir, os.path.join(el.name.split(".")[0] + ".csv"))

            # Create image object and read annotation data
            oImgCur = cv2.imread(el.path)
            lAnnotations = readAnnotationData(lblPath)

            # Transform the data
            dataTrans = transformData(oImgCur, lAnnotations, imgCrpDims, vProb, nAugments)

            # Save the transformed data
            saveAugmentedData(dataTrans, targetDir, el.name.split(".")[0])

            # Inplace replacement
            if targetDir == imgDir:
                os.remove(el.path)

            if targetDir == lblDir:
                os.remove(lblPath) 




if __name__ == "__main__":
    lblDir = r"Tensorflow/workspace/images/Test"
    imgDir = r"Tensorflow/workspace/images/Test"
    targetDir = r"Tensorflow/workspace/images/Test"
    augmentData(imgDir, lblDir, targetDir, [320,320], 10, [0.5, 0.5, 0.25, 0.25], "png")



  