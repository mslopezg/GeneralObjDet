from od_api import ObjDetAPI
import sys
import os
if __name__ == "__main__":
    flag = True
    api = ObjDetAPI()
    if len(sys.argv) >= 1:
        if 'install' in sys.argv:
            os.system('pip install --ignore-installed --upgrade tensorflow==2.5.0 tensorflow-gpu==2.5.0')
            os.system('pip install matplotlib')
            os.system('pip install pyyaml')
            os.system('pip install opencv-python')
            os.system('pip install cython')
            os.system('pip install pytz')
            os.system('pip install gin-config')
            os.system('pip install tensorflow_addons')
        if 'skip' in sys.argv:
            flag = False # if True, skip workspace, repo, and protoc
        if sys.argv[1] == 'GPU':
            api.verify_gpu()
            sys.exit()
    if flag:
        # create folders for workspace
        api.verify_workspace()
        # install tensorflow object detection repo if not already present
        api.verify_tensorflow_repo()
        # verify protoc installation
        api.verify_protoc()
    # verify set up
    api.verify_environment_setup()
    print('\n\n\nPlace images with annotations in Tensorflow/workspace/<MODEL_NAME>/images/ as train and test folders\nDownload model, extract contents, and place checkpoint/ saved_model/ and pipeline.config in the pre-trained-models/ \n\n\n')
    print('if OK: continue \n\nNo module found errors run: python setup.py install \n installs necessary packages\n\nRun: python setup.py skip \n skips folder creation, cloning repo, and protoc\n\nRun: python setup.py GPU\n verifies GPU')