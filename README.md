# GeneralObjDet
General Object Detection Model Repo


# GeneralObjDet Installation Walkthrough
<p>To train your own custom General Object Detection model, follow these steps: </p>
<br />
<p>For a more in depth tutorial on environment setup, particularly involving path variables on windows or CUDA and CUDNN installation, go to https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html</p>

## Steps
<br />
<b>Step 1.</b> Clone this repository: https://github.com/mslopezg/GeneralObjDet
<br/><br/>
<b>Step 2.</b> Create a new virtual environment:
<pre>
conda create -n tensorflow pip python=3.9
</pre> 
<br/>
If fail, Anaconda Python 3.9 is not installed
<br/>
<b>Step 3.</b> Activate your virtual environment:
<pre>
conda activate tensorflow
</pre>
<br/>
<b>Step 4.</b> Edit the top of \GeneralObjDet\src\od_api.py
<pre>CUSTOM_MODEL_NAME = 'custom_model_name' # change to liking
PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8' # change based on download
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz' # change based on download
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py' # change script if not generating annotations using labelImg on step X
labels = [{'name':'class1', 'id':1},{'name':'class2', 'id':2} ] # change depending on data
</pre>
<br/>
<b>Step 5.</b> Inside \GeneralObjDet\src run:
<pre>python setup.py install
</pre>
If there is a problem or would like more information on setup, for example: GPU setup, call: 
<pre>python setup.py help
</pre>
<br/>
<b>Step 6.</b> Place images in train and test folders and collect annotations
<pre>
\GeneralObjDet\Tensorflow\workspace\images\train
\GeneralObjDet\Tensorflow\workspace\images\test
</pre>
Use labelImg to annotate the images, place annotations in same folder (will be .xml files if labelImg is used)
<pre>pip install labelImg
</pre>
Change generate tf records script if using a different method of annotations.
<b>Step 7.</b>Obtain model
<br/>
Download model from <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">Tensorflow 2 ModelZoo</a>, extract contents, and place model folder in
<pre>
\GeneralObjDet\Tensorflow\workspace\pre-trained-models\custom_downloaded_model
</pre>
<br/>
<b>Step 8.</b> Set up training
<br />
Inside \GeneralObjDet\src run:
<pre>python train.py -l -r -c -e 2000 -b 2 -s
# -l creates labelmap based on definition inside od_api.py
# -r creates tf_records based on TF_RECORD_SCRIPT_NAME in od_api.py
# -c copies pretrained model config file to \GeneralObjDet\Tensorflow\workspace\custom_model_name\ and prepares it for training
# -e specifies how many epochs to train for
# -b specifies the batch size
# -s exits the program before training, useful to verify setup before actually training.
</pre>
If there is a problem or would like more information on train setup, call: 
<pre>python train.py help
</pre>
<br />
<b>Step 9.</b> Train the model
<pre>python train.py
</pre>
<br /> <br/>
<b>(Optional) Step 10.</b>Evaluate the model
<br />
Inside \GeneralObjDet\src run:
<pre>python test_single.py -c ckpt-1 -i image_path -s save_path -d -n 10 -t 0.7
# -c specifies checkpoint to load
# -i specifies image path
# -s specifies path to save result
# -d specifies if images are to be displayed with cv2
# -n specifies the number of boxes to detect on the image
# -t specifies the threshold for boxes to be displayed
</pre>
If there is a problem or would like more information on test setup, call: 
<pre>python test_single.py help
</pre>
To evaluate a directory run:
<pre>python test_batch.py -c ckpt-1 -i image_path -s save_path -d -dn 5 -n 10 -t 0.7
# -c specifies checkpoint to load
# -i specifies image path
# -s specifies path to save result
# -d specifies if images are to be displayed with cv2
# -dn specifies the number of images to display
# -n specifies the number of boxes to detect on the image
# -t specifies the threshold for boxes to be displayed
</pre>
If there is a problem or would like more information on train setup, call: 
<pre>python test_batch.py help
</pre>
