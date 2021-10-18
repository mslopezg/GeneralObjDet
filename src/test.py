from od_api import ObjDetAPI
import sys
import os
import tensorflow as tf
if __name__ == "__main__":
    api = ObjDetAPI()
    if len(sys.argv) == 5:
        model = api.load_model(sys.argv[sys.argv.index('-c')+1])
        api.detect_batch(model, sys.argv[sys.argv.index('-i')+1])
    sys.exit()