import subprocess

# Paths for Test partition
xmldirTest = r"Tensorflow/workspace/images/test"
labelsPathTest = r"Tensorflow/workspace/annotations/labelmap.pbtxt"
outputPathTest = r"Tensorflow/workspace/annotations/test.record"
imageDirTest = r"Tensorflow/workspace/images/test"

# Create tfrecord for test patition
cmd = "py Tensorflow/scripts/preprocessing/generate_tfrecord.py -x" + xmldirTest + " -l " + labelsPathTest
cmd += " -o " + outputPathTest + " -i " + imageDirTest
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
out, err = p.communicate()
print(out)
print(err)

# Paths for Training partition
xmldirTrain = r"Tensorflow/workspace/images/train"
labelsPathTrain = r"Tensorflow/workspace/annotations/labelmap.pbtxt"
outputPathTrain = r"Tensorflow/workspace/annotations/train.record"
imageDirTrain = r"Tensorflow/workspace/images/train"

# Create tfrecord for training partition
cmd = "py Tensorflow/scripts/preprocessing/generate_tfrecord.py -x" + xmldirTrain + " -l " + labelsPathTrain
cmd += " -o " + outputPathTrain + " -i " + imageDirTrain

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell = True)
out, err = p.communicate()
print(out)
print(err)
