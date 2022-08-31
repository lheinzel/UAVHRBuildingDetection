
import albumentations as alb
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    coords = list(bbox)
    for i, el in enumerate(coords):
        coords[i] = int(el)

    x_min, y_min, x_max, y_max = coords

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes):
    img = image.copy()
    for entry in bboxes:
        bbox = entry[0:4]
        class_name = entry[-1]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)




def readAnnotationData(lblPath):
    dfLblData = pd.read_csv(lblPath, sep=";")
    listAnnotation = dfLblData[['xmin' , 'ymin', 'xmax', 'ymax', 'class']].values.tolist()
    return listAnnotation


def transformData(oAugmenter, oImage, lAnnotation, nAugments):
    dataTransformed = [];

    for i in range(nAugments):
        dataTransCur = oAugmenter(image=oImage, bboxes=lAnnotation)
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

        # Transform bbox dimensions to int
        dfAnnotations[["xmin", "ymin", "xmax", "ymax"]] = dfAnnotations[["xmin", "ymin", "xmax", "ymax"]].astype(int)

        # Write Annotation data
        dfAnnotations.to_csv(os.path.join(targetDir, lblNameCur), sep=";", index=None)

        # Write image
        cv2.imwrite(os.path.join(targetDir, imgNameCur), dat["image"])




def augmentData(oAugmenter, imgDir, lblDir, targetDir, nAugments, imgFileExt, bVisualize=False):
    for el in os.scandir(imgDir):
        if el.is_file() and el.name.split(".")[1] == imgFileExt:
            lblPath = os.path.join(lblDir, os.path.join(el.name.split(".")[0] + ".csv"))

            # Create image object and read annotation data
            oImgCur = cv2.imread(el.path)
            lAnnotations = readAnnotationData(lblPath)

            # Transform the data
            dataTrans = transformData(oAugmenter, oImgCur, lAnnotations, nAugments)

            if bVisualize:
                for dat in dataTrans:
                    if dat["bboxes"]:
                        visualize(dat["image"], dat["bboxes"])
                        pass

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
    imgCrpDims = [320, 320]

    bboxParams = alb.BboxParams(format='pascal_voc', min_visibility=0.75)
    oAugmenter = alb.Compose([alb.RandomCrop(imgCrpDims[0],imgCrpDims[1]),
                            alb.ColorJitter(hue=0.1, brightness=0.5, saturation=0.5, p=0.5),
                            alb.HorizontalFlip(p=0.25),
                            alb.VerticalFlip(p=0.25)], bboxParams)

    augmentData(oAugmenter, imgDir, lblDir, targetDir, 10, "png")



  