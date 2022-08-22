import subprocess

xmldir = r"Tensorflow/workspace/images/test"
labelsPath = r"Tensorflow/workspace/training_Demo/annotations/labelmap.pbtxt"
outputPath = r"Tensorflow/workspace/training_Demo/annotations/test.record"
imageDir = r"Tensorflow/workspace/images/test"

cmd = "py Tensorflow/scripts/preprocessing/generate_tfrecord.py -x" + xmldir + " -l " + labelsPath + " -o " + outputPath + " -i " + imageDir
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell = True)
out, err = p.communicate()
