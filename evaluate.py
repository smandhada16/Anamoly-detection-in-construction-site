"""
Script to eveluation model
Usage:
python evaluate.py --dataset /path/to/folder/images --checkpoint /point/to/model
Example:
python evaluate.py --dataset /dataset/val --checkpoint /box20190222T1237/mask_rcnn_box_0019.h5
"""
import json
import os
import sys
import time
import cv2
import pickle
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT_DIR = os.path.abspath("../../")

# To find local version of the library
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
from mrcnn import utils
from pathlib import Path
from samples.workersDetection1 import workersDetection1
from collections import namedtuple, defaultdict, deque, Counter
from sklearn.metrics import confusion_matrix, classification_report
import itertools

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from tqdm import tqdm
from keras import backend as K
#import tensorflow.python.keras.backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#sess=tf.compat.v1.keras.backend.get_session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
#tf.compat.v1.keras.backend.get_session()


ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    help="Path to dataset folder that contained json annotation file("
    "via_region_data.json)",
    required=True,
)
ap.add_argument("-c", "--checkpoint", help="Path to checkpoint", required=True)
ap.add_argument(
    "-m",
    "--mode",
    help="CPU or GPU, default is CPU (/cpu:0 or /gpu:0)",
    default="/cpu:0",
)
args = vars(ap.parse_args())

# trained weights
MODEL_WEIGHTS_PATH = args["checkpoint"]
DATASET_DIR = args["dataset"]
DEVICE = args["mode"]

Detection = namedtuple("Detection", ["gt_class", "pred_class", "overlapscore"])
## dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
def dice_coef(y_true, y_pred):
    intersec = y_true * y_pred
    union = y_true + y_pred
    if intersec.sum() == 0:
        dice_coef = 0
    else:
        dice_coef = round(intersec.sum() * 2 / union.sum(), 2)
    return dice_coef

def coeff_per_image(metric_name, image_id, pred, gt_mask, gt_class_id):
    coeff_dict = {}
    CLASS_NUMBER = [1,2,3,4]
    for clsid in CLASS_NUMBER:
        coeff_dict[clsid] = []
        gt_index = np.where(gt_class_id == clsid)

        # if there is no groundtruth or no predicted mask, the coefficient is equal to zero
        if gt_index[0].size == 0 or len(pred['masks']) == 0:
            coeff_dict[clsid].append(0)
        else:
            # get the union of all groundtruth masks belong to clsid
            gt_mask_per_class = gt_mask[:, :, gt_index[0]]  # get groundtruth mask

            _gt_sum = np.zeros((gt_mask.shape[0], gt_mask.shape[1]))

            for gt_num in range(gt_mask_per_class.shape[2]):  # as there may be over one mask per class
                _gt = gt_mask_per_class[:, :, gt_num]
                _gt_sum = _gt_sum + _gt

            _gt_union = (_gt_sum > 0).astype(int)

            # get the union of all predicted masks belong to clsid
            pred_index = np.where(pred['class_ids'] == clsid)
            pred_mask_per_class = pred['masks'][:, :, pred_index[0]]

            _mask_sum = np.zeros((pred['masks'].shape[0], pred['masks'].shape[1]))

            for num in range(pred_mask_per_class.shape[2]):
                _mask = pred_mask_per_class[:, :, num]
                _mask_sum = _mask_sum + _mask

            _mask_union = (_mask_sum > 0).astype(int)

            if metric_name == 'jaccard index':
                coeff_dict[clsid].append(jaccard_coef(_mask_union, _gt_union))
            elif metric_name == 'dice':
                coeff_dict[clsid].append(dice_coef(_mask_union, _gt_union))

    return coeff_dict

#kernel = np.ones((3, 3), np.uint8)
def compute_batch_detections(model, image_ids):
    # Compute VOC-style Average Precision
    APs = []
    PRs = []
    REs = []
    detections = []
    ARs = []

    
    dice_dic = {}

    for image_id in tqdm(image_ids):
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id, use_mini_mask=False
        )        
        w, h = image.shape[1], image.shape[0]
        if w < h:
            w = h
        else:
            h = w
        #image = cv2.addWeighted(image, alpha, np.zeros(img.shape, img.dtype), 0, beta)
        #image = cv2.resize(image, (w, h),interpolation=cv2.INTER_CUBIC)    
        
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]

        AP, precisions, recalls, overlaps = utils.compute_ap(
            gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
            iou_threshold=0.4, 
        )

        APs.append(AP)
        PRs.append(precisions)
        REs.append(recalls)
        AR, _ = utils.compute_recall(r["rois"], gt_bbox, iou=0.5) 
        ARs.append(AR)
        

        # list_overlaps
        detection = Detection(gt_class_id, r, overlaps)

        detections.append(detection)
        dice_dic[image_id] = coeff_per_image('dice', image_id, r, gt_mask, gt_class_id)
    try:
        print("[INFO] Dice Coefficient: ", )
        dice_path = os.getcwd() + "dice_coeff.p"
        pickle.dump(dice_dic, open(dice_path, 'wb'))
        print(dice_dic)
    except Exception:
        pass

    return detections, APs, PRs, REs,ARs


def inspect_class_predicted(result_detection):
    inspect_class = defaultdict(int)
    threshold = 0.4
    class_names = dataset.class_names

    y_pred_name = []
    y_true_name = []

    for detect in tqdm(result_detection):

        y_index_pred = []
        y_index_true = []
        index_pred = []
        index_ground_tr = []

        gt_class_ids = detect.gt_class
        pred_class_ids = detect.pred_class["class_ids"]
        overlaps = detect.overlapscore

        gt_class_ids = gt_class_ids[gt_class_ids != 0]
        pred_class_ids = pred_class_ids[pred_class_ids != 0]

        for i, j in itertools.product(
            range(overlaps.shape[0]), range(overlaps.shape[1])
        ):

            if overlaps[i, j] > threshold:
                index_pred.append(i)
                index_ground_tr.append(j)
                if gt_class_ids[j] == pred_class_ids[i]:
                    inspect_class[class_names[gt_class_ids[j]]] += 1

        for i, j in zip(index_pred, index_ground_tr):
            y_index_pred.append(pred_class_ids[i])
            y_index_true.append(gt_class_ids[j])

        for i, j in zip(y_index_true, y_index_pred):
            y_true_name.append(class_names[i])
            y_pred_name.append(class_names[j])

    return inspect_class, y_true_name, y_pred_name


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    print("\n[INFO] Confusion Matrix")

    columnwidth = max([len(x) for x in labels] + [5])
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


config = workersDetection1.WorkersDetectionConfig()


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.33
    ROI_POSITIVE_RATIO = 0.33
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    BACKBONE = "resnet101"    
    #IMAGE_MAX_DIM = 1920
    #POST_NMS_ROIS_INFERENCE = 1500
    RPN_NMS_THRESHOLD = 0.8



    """ THIS CONFIGURATION GAVE mAP: 0.245
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.66
    ROI_POSITIVE_RATIO = 0.66
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    BACKBONE = "resnet101"    
    #IMAGE_MAX_DIM = 1920
    POST_NMS_ROIS_INFERENCE = 5000
    PRE_NMS_LIMIT = 8000    
    """


if __name__ == "__main__":
    start = time.time()
    config = InferenceConfig()
    config.display()

    # Load validation dataset    
    dataset = workersDetection1.WorkersDetectionDataset()
    dataset.load_workers(DATASET_DIR,"val")
    dataset_dir = os.path.join(DATASET_DIR, "val")

    # Must call before using the dataset
    dataset.prepare()

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(
            mode="inference", model_dir=MODEL_WEIGHTS_PATH, config=config
        )

    # Load weights
    print("[INFO] Loading weights ", MODEL_WEIGHTS_PATH)
    model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)

    result_detection, APs, precisions, recalls, ARs = compute_batch_detections(
        model, dataset.image_ids
    )

    print("Number of Images: {}\nClasses: {}".format( len(dataset.image_ids), dataset.class_names[:] )   )
    
    #print(f"[INFO] mAP IoU=50: ", np.mean(APs))
    mAP=np.mean(APs)
    mAR=np.mean(ARs)
    f_score = (2 * mAP * mAR)/(mAP + mAR)
    print("[INFO] mAP: ", mAP)
    print("[INFO] mRE: ", mAR)
    print("[INFO] f score: ",f_score)
    

    inspect_class, name_true, name_pred = inspect_class_predicted(result_detection)
    print(
        "\n[INFO] The Total Number Of Annotations Model Could Predict: ", len(name_pred)
    )

    print(
        "\n[INFO] The Total Number Of Annotations Model Predicted Correctly: ", sum(inspect_class.values())
    )
    print("[INFO] Number of correct predicted for each classes")
    # for k, v in inspect_class.items():
    #     print("[INFO] {}: {}".format(k,v))
    class_names = ["full", "half-empty", "empty", "obstacle"]
    for name in class_names:
        try:
            print("[INFO] {}: {}".format(name, inspect_class[name]))
        except:
            print("[INFO] {}: 0".format(name))

    """path = Path(DATASET_DIR)
    for jfile in path.glob("*.json"):
        with open(str(jfile), "r") as f:
            json_file = json.load(f)"""
            
    json_file=json.load(open(os.path.join(dataset_dir, "training.json")))

    total_annotation = 0
    classes_count = deque()
    for key, value in json_file.items():
        total_annotation += len(value["regions"])
        for region in value["regions"]:
            try:
                classes_count.append(region["region_attributes"]["class"])
            except:
                continue

    print("\n[INFO] The Total Annotations Of Ground Truth: {}".format(total_annotation))
    count_class =  Counter(classes_count)
    for name in class_names:
        try:
            print("[INFO] {}: {}".format(name, count_class[name]))
        except:
            print("[INFO] {}: 0".format(name))

    # for name, value in Counter(classes_count).most_common():
    #     print("[INFO] {}: {}".format(name, value))

    #my_labels = ["full", "empty", "obstacle", "half-empty"]
    
    print("\n")
    try:
        cm = confusion_matrix(name_true, name_pred, labels=class_names)
        print_cm(cm, class_names)
        print("\nClassification Report\n")
        print(classification_report(name_true, name_pred))
    except:
        print("None!")
    

    execution_time =  (time.time() - start) / 60
    print("\n[INFO] Execution Time: {} minutes".format(execution_time))
