# GeneralObjDet
General Object Detection Model Repo


# GeneralObjDet Installation Walkthrough
<p>To train your own custom General Object Detection model, follow these steps: 

## Steps
<br />
<b>Step 1.</b> Clone this repository: https://github.com/mslopezg/GeneralObjDet
<br/><br/>
<b>Step 2.</b> Create a new virtual environment 
<pre>
python -m venv tensorflow
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source tensorflow/bin/activate # Linux
.\tensorflow\Scripts\activate # Windows 
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
</pre>
<br/>
<b>Step 5.</b> Inside \GeneralObjDet\src run ```python setup.py install```, call ```python setup.py help``` if there is a problem. 
<br/>
<b>Step 6.</b> Manually divide collected images into two folders train and test. So now all folders and annotations should be split between the following two folders. <br/>
\GeneralObjDet\Tensorflow\workspace\images\train<br />
\GeneralObjDet\Tensorflow\workspace\images\test
<br/><br/>
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