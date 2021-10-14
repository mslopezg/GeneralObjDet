from od_api import ObjDetAPI
import sys
import os
import tensorflow as tf
if __name__ == "__main__":
    api = ObjDetAPI()
    if len(sys.argv) >= 1:
        if 'labelmap' in sys.argv:
            api.create_label_map()
        if 'records' in sys.argv:
            api.create_tf_records()
        if 'config' in sys.argv:
            api.update_config()
        if 'steps' in sys.argv:
            num_steps = sys.argv[sys.argv.index('steps')+1]
            api.train(num_steps)
        else:
            api.train()
        sys.exit()