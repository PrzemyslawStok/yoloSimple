#convert_weights.py
import numpy as np

from convert_weights import load_weights
from yolov3 import YOLOv3Net
from yolov3 import parse_cfg

def main():
    weightfile = "weights/yolov3.weights"
    cfgfile = "cfg/yolov3.cfg"
    model_size = (416, 416, 3)
    num_classes = 80
    model = YOLOv3Net(cfgfile, model_size, num_classes)
    load_weights(model, cfgfile, weightfile)
    try:
        model.save_weights('weights/yolov3_weights.tf')
        print('\nThe file \'yolov3_weights.tf\' has been saved successfully.')
    except IOError:
        print("Couldn't write the file \'yolov3_weights.tf\'.")

main()