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
<b>Step 2.</b> Create a new virtual environment 
<pre>
conda create -n tensorflow pip python=3.9
</pre> 
<br/>
If fail, Anaconda Python 3.9 is not installed
<b>Step 3.</b> Activate your virtual environment
<pre>
conda activate tensorflow
</pre>
<br/>
<b>Step 4.</b> Inside \GeneralObjDet\src run 
<pre>python setup.py install
</pre>
If there is a problem, call 
<pre>python setup.py help
</pre>
<br/>
<b>Step 5.</b> Place images with annotations in Tensorflow/workspace/model_name/images/ as train and test folders
<pre>
\GeneralObjDet\Tensorflow\workspace\images\train
 <br/>
\GeneralObjDet\Tensorflow\workspace\images\test
</pre>
Download model from <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md">Tensorflow 2 ModelZoo</a>, extract contents, and place model folder in the pre-trained-models/ folder
<br/>
<b>Step 7.</b> Begin training process by open
<br /><br/>
<b>Step 8.</b> During this process the Notebook will install Tensorflow Object Detection. You should ideally receive a notification indicating that the API has installed successfully at Step 8 with the last 
If not, resolve installation er
<br /> <br/>
<b>Step 9.</b> Once you get to step 6. Train the model, inside of the notebook, you may choose to train the model from within the notebook. I have noticed however that training inside of a separate terminal on  
<br />
<b>Step 10.</b> You can optionally evaluate your model inside of Tensorboard. Once the model has been trained and you have run the evaluation command under Step 7. Navigate to the evaluation folder for your trained model e.g. 
<pre> cd Tensorlfow/workspace/models/my_ssd_mobnet/eval</pre> 
and open Tensorboard with the following command
<pre>tensorboard --logdir=. </pre>
Tensorboard will be accessible through your browser and you will be able to see metrics including mAP - mean Average Precision, and Recall.
<br />