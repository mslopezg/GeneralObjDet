from od_api import ObjDetAPI
import sys
import os
if __name__ == "__main__":
    if 'help' in sys.argv:
        print('Example call:\n \
            python setup.py <flags>\n \
            \n install \t full initial install, called by itself \
            \n protoc \t verifies protoc installation \
            \n workspace \t creates workspace (folders) \
            \n tf_repo \t verifies if Tensorflow 2 repo is cloned \
            \n rebuild \t rebuilds Tensorflow 2 repo \
            \n skip \t skips full initial install, only verifies object detection api is built \
            \n GPU \t verifies if GPU is being used \
            \n\n if no module found error, \'pip install <module_name>\'')
        sys.exit()

    api = ObjDetAPI()
    if 'install' in sys.argv:
        os.system('pip install --ignore-installed --upgrade tensorflow==2.5.0')
        os.system('pip install cython')
        os.system('pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI') # might need changing for eval script
        os.system('pip install wget')
        # create folders for workspace
        api.verify_workspace()
        # install tensorflow object detection repo if not already present
        api.verify_tensorflow_repo()
        # retrieve pretrained model from url
        api.retrieve_pretrained()
        # verify protoc installation
        api.verify_protoc()
        # verify set up
        api.verify_environment_setup()

    if 'protoc' in sys.argv:
        api.verify_protoc()
        print('If a problem arises, call \'python setup.py help\' or visit README.md in GeneralObjDet github repo')
        sys.exit()

    if 'workspace' in sys.argv:
        api.verify_workspace()
        print('If a problem arises, call \'python setup.py help\' or visit README.md in GeneralObjDet github repo')
        sys.exit()

    if 'tf_repo' in sys.argv:
        api.verify_tensorflow_repo()
        print('If a problem arises, call \'python setup.py help\' or visit README.md in GeneralObjDet github repo')
        sys.exit()

    if 'rebuild' in sys.argv:
        api.rebuild_obj_det_api()
        print('If a problem arises, call \'python setup.py help\' or visit README.md in GeneralObjDet github repo')
        sys.exit()

    if 'skip' in sys.argv:
        api.verify_environment_setup()

    if 'GPU' in sys.argv:
        api.verify_gpu()
        print('If a problem arises, call \'python setup.py help\' or visit README.md in GeneralObjDet github repo')
        sys.exit()

    print('\n\n\nPlace images with annotations in Tensorflow/workspace/<MODEL_NAME>/images/ as train and test folders\n \
    Download model, extract contents, and place model folder in the pre-trained-models/ \n')

    print('if OK: continue\n \
            No module found errors run: python setup.py install \n \
             installs necessary packages\n \
            Run: python setup.py skip \n \
             skips folder creation, cloning repo, and protoc \n \
            Run: python setup.py GPU\n \
             verifies GPU')
    print('If a problem arises, call \'python setup.py help\' or visit README.md in GeneralObjDet github repo')
    sys.exit()