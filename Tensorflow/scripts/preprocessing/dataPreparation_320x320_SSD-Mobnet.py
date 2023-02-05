from datasetPartitioning import partition
from dataAugmentation import augmentData
import albumentations as alb
from shutil import rmtree
from shutil import copyfile
import os
import tarfile

imgSource = r"Datasets/UAVHighRiseBuildingsKorea/DataCropped_320x320/Images";
lblSource = r"Datasets/UAVHighRiseBuildingsKorea/DataCropped_320x320/Labels";
destPath = r"Tensorflow/workspace/images"
annotationPath = r"Tensorflow/workspace/annotations"
lblMapPath = r"Tensorflow/workspace/annotations/labelmap.pbtxt"
lblMapSrcPath = r"Datasets/UAVHighRiseBuildingsKorea/labelmap.pbtxt"
cmpAnnotationPath = r"Tensorflow/workspace/annotations.tar.gz"
ratioTest = 0.2
ratioValid = 0.2
INPUT_DIMS = [320,320];


# Copy labelmap from dataset
if not os.path.exists(lblMapPath):
  copyfile(lblMapSrcPath, lblMapPath);

# Partiion the dataset
if not os.path.exists(destPath) or os.listdir(destPath) == []:
  partition(imgSource, lblSource, destPath, ratioTest, ratioValid,"png")

  nAugmentations = 10
  testPath = os.path.join(destPath,"test")
  trainPath = os.path.join(destPath, "train")
  validPath = os.path.join(destPath, "valid")

  bboxParams = alb.BboxParams(format='pascal_voc', min_visibility=0.75)
  oAugmenter = alb.Compose([alb.RandomCrop(INPUT_DIMS[0],INPUT_DIMS[1]),
                          #alb.Affine(shear=[-2.5,2.5], rotate=[-2.5 ,2.5], fit_output=True, p=0.5), 
                          #alb.Resize(INPUT_DIMS[0]+50,INPUT_DIMS[1]+50),
                          #alb.CenterCrop(INPUT_DIMS[0],INPUT_DIMS[1]),
                          alb.ColorJitter(hue=0.1, brightness=0.5, saturation=0.5, p=0.5),
                          alb.HorizontalFlip(p=0.25),
                          alb.VerticalFlip(p=0.25)], bboxParams)

  # For the test data: only crop the oversize images randomply
  augmentData(alb.Compose([alb.RandomCrop(INPUT_DIMS[0],INPUT_DIMS[1])],bboxParams),
              testPath, testPath, testPath, 1, "png");

  # Augment Train data
  augmentData(oAugmenter, trainPath, trainPath, trainPath, nAugmentations, "png")

  # Augment Validation data
  if os.path.exists(validPath):
    augmentData(oAugmenter, validPath, validPath, validPath, nAugmentations, "png")


# Create tfrecord files
os.system("py Tensorflow/scripts/preprocessing/generate_tfrecord.py -x" + os.path.join(destPath,"train") + " -l " +
          lblMapPath + " -o " + os.path.join(annotationPath,"train.record") + " -i " + os.path.join(destPath, "train"))

os.system("py Tensorflow/scripts/preprocessing/generate_tfrecord.py -x" + os.path.join(destPath,"test") + " -l " +
          lblMapPath + " -o " + os.path.join(annotationPath,"test.record") + " -i " + os.path.join(destPath, "test"))

os.system("py Tensorflow/scripts/preprocessing/generate_tfrecord.py -x" + os.path.join(destPath,"valid") + " -l " +
          lblMapPath + " -o " + os.path.join(annotationPath,"valid.record") + " -i " + os.path.join(destPath, "valid"))

# Compress anntotations directory to tar file
if not os.path.exists(cmpAnnotationPath):
  with tarfile.open(cmpAnnotationPath, "w:gz") as tar:
    tar.add(annotationPath, arcname=os.path.basename(annotationPath))