from od_api import ObjDetAPI
import sys
if __name__ == "__main__":
    if 'help' in sys.argv:
        print('Example call:\n \
            python train.py -l -r -c -e 2000 -b 2 -s \t This will prepare training but exit before train start so that the user can verify setup\n \
            after training setup complete run: \t python train.py  \
            \n -l \t labelmap flag (Required Once) \
            \n -r \t tf records flag (Required Once) \
            \n -c \t update pipeline.config flag (Required Once) \
            \n -s \t skip training, (Optional) \
            \n -e \t epochs flag, default: 2000 (Optional) \
            \n -b \t batch_size flag, default: 2000 (Optional) \
            \n\nIf -e and -b flags are in call, -c must also be included \
            \nThere must be a pipeline.config file inside Tensorflow/workspace/models/<model_name>/ folder for training to occur')
        sys.exit()
    api = ObjDetAPI()
    if '-l' in sys.argv:
        api.create_label_map()
    if '-r' in sys.argv:
        api.create_tf_records()
    if '-c' in sys.argv:
        if '-e' in sys.argv:
            num_steps = sys.argv[sys.argv.index('-e')+1]
        if '-b' in sys.argv:
            batch_size = sys.argv[sys.argv.index('-b')+1]
        api.update_config(batch_size = batch_size,num_steps= num_steps)
    if '-s' in sys.argv:
        sys.exit()
    else:
        api.train()
    print('if any problems, call \'python train.py help\'')
    sys.exit()