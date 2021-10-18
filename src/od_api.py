import os
import sys

CUSTOM_MODEL_NAME = 'ducky_detector' # change   
PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8' # change depending on chosen model below
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz' # coordinate with above
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

labels = [{'name':'duck', 'id':1}]


class ObjDetAPI:
    def __init__(self):
        pass

    def verify_workspace(self):
        # creates workspace if not already done
        for path in paths.values():
            if not os.path.exists(path):
                if os.name == 'posix':
                    os.system("mkdir -p "+ path)
                if os.name == 'nt':
                    os.system("mkdir " +  path)

    def verify_tensorflow_repo(self):
        # clones tensorflow models repo
        if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
            os.system("git clone https://github.com/tensorflow/models " + paths['APIMODEL_PATH'])

    def verify_protoc(self):
        if os.name=='nt':
            try: 
                import wget
            except ModuleNotFoundError:
                os.system("pip install wget")
                import wget
            

        if os.name=='posix':  
            os.system("apt-get install protobuf-compiler")
            os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .")

        if os.name=='nt':
            url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
            wget.download(url)
            os.system("move protoc-3.15.6-win64.zip " + paths['PROTOC_PATH'])
            os.system("cd " + paths['PROTOC_PATH'] + " && tar -xf protoc-3.15.6-win64.zip")
            os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
            os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python -m pip install --use-feature=2020-resolver .")
            os.system("cd Tensorflow/models/research/slim && pip install -e . ")
            print("\n\nadd " + paths['PROTOC_PATH'] +  "\\bin to Path environment variables")
            sys.exit()

    def verify_environment_setup(self):
        os.system("python "+ os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py'))
        import cv2        

    def verify_gpu(self):
        import tensorflow as tf
        print(tf.reduce_sum(tf.random.normal([1000, 1000])))

    def create_label_map(self):
        with open(files['LABELMAP'], 'w') as f:
            for label in labels:
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')

    def create_tf_records(self):
        #TODO data generator
        # currently this is using the general tf record script
        os.system("python generate_tfrecord.py -x "+os.path.join(paths['IMAGE_PATH'], 'train')+" -l "+files['LABELMAP']+" -o "+os.path.join(paths['ANNOTATION_PATH'], 'train.record'))
        os.system("python generate_tfrecord.py -x "+os.path.join(paths['IMAGE_PATH'], 'test')+" -l "+files['LABELMAP']+" -o "+os.path.join(paths['ANNOTATION_PATH'], 'test.record')) 
        

    def update_config(self):
        import tensorflow as tf
        from object_detection.utils import config_util
        from object_detection.protos import pipeline_pb2
        from google.protobuf import text_format

        # copy the pipeline.config from pretrained model to the workspace model folder
        if os.name =='posix':
            os.system("cp "+ os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config') + " " + os.path.join(paths['CHECKPOINT_PATH']))
        if os.name == 'nt':
            os.system("copy " + os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config') + " " + os.path.join(paths['CHECKPOINT_PATH']))

        # read and edit pipeline.config
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
            proto_str = f.read()                                                                                                                                                                                                                                          
            text_format.Merge(proto_str, pipeline_config) 
        pipeline_config.model.ssd.num_classes = len(labels)
        pipeline_config.train_config.batch_size = 4
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

        # write edited file to pipeline.config
        config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
        with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
            f.write(config_text)

    def train(self, num_steps= None):
        import wget
        import object_detection
        import tensorflow as tf
        from google.protobuf import text_format
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        from object_detection.utils import config_util
        from object_detection.protos import pipeline_pb2
        from matplotlib import pyplot as plt
        import numpy as np
        import matplotlib

        TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
        # num_train_steps = epochs
        if num_steps:
            command = "python "+TRAINING_SCRIPT+" --model_dir="+paths['CHECKPOINT_PATH']+" --pipeline_config_path="+files['PIPELINE_CONFIG']+" --num_train_steps="+str(num_steps) # train steps edited here
        else:
            command = "python "+TRAINING_SCRIPT+" --model_dir="+paths['CHECKPOINT_PATH']+" --pipeline_config_path="+files['PIPELINE_CONFIG']
        os.system(command)

    
    def load_model(self, ckpt_path):
        # ckpt is the latest checkpoint from trained model, string
        # example: 'ckpt-6'
        import os
        import tensorflow as tf
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        from object_detection.utils import config_util

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], ckpt_path)).expect_partial()

        return detection_model

    def detect_image(self,detection_model,image_path):
        import cv2 
        import numpy as np
        from matplotlib import pyplot as plt
        import os
        import tensorflow as tf
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        from object_detection.utils import config_util
        

        # display = False, no images shown
        # display = True, images shown

        @tf.function
        def detect_fn(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections

        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        img = cv2.imread(image_path)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=20,
                    min_score_thresh=.1,
                    agnostic_mode=False)

        cv2.imshow('image',image_np_with_detections)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_batch(self,detection_model,dir_path, save_path= None, display = True):
        # TODO: function to load batch, list of images, and perform detection
        # savepath saves to folder
        import cv2 
        import numpy as np
        from matplotlib import pyplot as plt
        import os
        import tensorflow as tf
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        from object_detection.utils import config_util
        

        # display = False, no images shown
        # display = True, images shown

        @tf.function
        def detect_fn(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections

        detected_images = []
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        im_num = 0
        for filename in os.listdir(dir_path):
            if filename[-3:] == 'jpg':
                im_num += 1
                img = cv2.imread(dir_path+'/'+filename)
                image_np = np.array(img)

                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                detections = detect_fn(input_tensor)

                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                              for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                label_id_offset = 1
                image_np_with_detections = image_np.copy()

                viz_utils.visualize_boxes_and_labels_on_image_array(
                            image_np_with_detections,
                            detections['detection_boxes'],
                            detections['detection_classes']+label_id_offset,
                            detections['detection_scores'],
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=20,
                            min_score_thresh=.1,
                            agnostic_mode=False)
                if display:
                    cv2.imshow('image '+str(im_num),image_np_with_detections)
                detected_images.append(image_np_with_detections)
        if display:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_path:
            #TODO save images to save path
            pass
        
