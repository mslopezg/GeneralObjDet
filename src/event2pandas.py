import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf

def tabulate_events(dpath):
    for dname in os.listdir(dpath):
        ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        tags = ea.Tags()['tensors']
        out = []

        for tag in tags:
            if 'eval' not in tag:
                for event in ea.Tensors(tag):
                    out.append((tag,tf.make_ndarray(event.tensor_proto).item(0)))

        df = pd.DataFrame(out, columns = ['Metric', 'Value'])
        df.to_csv(dname+'.csv', index=False)


if __name__ == '__main__':
    path = r"C:\Users\Manuel Lopez\AppData\Roaming\SPB_Data\AccentureCV\GeneralObjDet\src\Tensorflow\workspace\models\ducky_detector_mobilenet\eval"
    tabulate_events(path)