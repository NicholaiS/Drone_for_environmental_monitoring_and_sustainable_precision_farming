import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
#from ShowImgs import show_imgs

iou_threshold = 0.5
      
def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def calculate_iou(gt_bbox, pred_bbox, threshold):
    tp = 0
    fp = 0
    fn = 0

    matched_indices = set()

    for i, pred_box in enumerate(pred_bbox):
        max_iou = 0
        max_idx = -1

        for j, gt_box in enumerate(gt_bbox):
            IoU = iou(pred_box, gt_box)

            if IoU > max_iou:
                max_iou = IoU
                max_idx = j

        if max_iou >= threshold and max_idx not in matched_indices:
            tp += 1
            matched_indices.add(max_idx)
        else:
            fp += 1

    fn = len(gt_bbox) - len(matched_indices)

    return tp, fp, fn

def F1(tp, fp, fn):
    if tp != 0 or fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp != 0 or fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if precision != 0 and recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1

# load ground truth boxes for all images
def load_ground_truth(gt_dir):
    gt_boxes = {}
    for filename in os.listdir(gt_dir):
        if filename.endswith('.xml'):
            tree = ET.parse(os.path.join(gt_dir, filename))
            root = tree.getroot()
            boxes = []
            for obj in root.findall('object'):
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
            gt_boxes[filename[:-4]] = boxes
    return gt_boxes

# load bounding box coordinates from a .txt file
def load_predictions(pred_dir):
    preds = {}
    for filename in os.listdir(pred_dir):
        if filename.endswith('.txt'):
            image_id = filename[:-4]
            with open(os.path.join(pred_dir, filename), 'r') as f:
                lines = f.readlines()
            boxes = []
            coords = []
            for line in lines:
                coords_pairs = line.strip().split(') (')
                coord = coords_pairs[0].replace('(', '').replace(')', '').split(', ')
                coords.append(coord)
                coord = coords_pairs[1].replace('(', '').replace(')', '').split(', ')
                coords.append(coord)
                    
                if len(coords) == 2:
                    tl_x, tl_y = map(int, coords[0])  # Access sublist directly
                    br_x, br_y = map(int, coords[1])  # Access sublist directly
                    boxes.append([tl_x, tl_y, br_x, br_y])
            preds[image_id] = boxes
    return preds

def calculate_ap(precision, recall):
    n = len(precision)
    auc = 0
    for i in range(1, n):
        auc += (recall[i] - recall[i-1]) * precision[i]
    return auc

# calculate precision-recall curve
gt_dir = 'Ground thruths'
pred_dir = 'Bounding boxes out\\80'
thresholds = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

# load ground truth boxes
gt_boxes = load_ground_truth(gt_dir)

# initialize lists to store precision and recall values
precision_list = []
recall_list = []

data_log = open("data log.txt", "w")

# loop over all thresholds
for threshold in thresholds:
    # load predicted boxes for the current threshold
    pred_boxes = load_predictions(os.path.join(pred_dir, str(threshold)))
    print(pred_boxes)

    # initialize variables to keep track of true positives, false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0

    # loop over all images and compute true/false positives/negatives
    for image_id, gt in gt_boxes.items():
        if image_id in pred_boxes:
            tp_image, fp_image, fn_image = calculate_iou(gt, pred_boxes[image_id], iou_threshold)
            tp += tp_image
            fp += fp_image
            fn += fn_image

    # compute precision, recall, and F1 score for the current threshold
    precision, recall, f1 = F1(tp, fp, fn)
    precision_list.append(precision)
    recall_list.append(recall)

    # Logging all the data for the different thresholds:
    data_log.write("Threshold: " + str(threshold)+ "\n")
    data_log.write("TP: " + str(tp)+ "\n")
    data_log.write("FP: " + str(fp)+ "\n")
    data_log.write("FN: " + str(fn)+ "\n")
    data_log.write("Precision: " + str(precision)+ "\n")
    data_log.write("Recall: " + str(recall)+ "\n")
    data_log.write("F1 score: " + str(f1)+ "\n")
    data_log.write("\n")
ap = calculate_ap(precision_list, recall_list)
data_log.write("Overall AP: " + str(ap)+ "\n")
data_log.close

# Plots the threshold values on the Precision-Recall curve:
for threshold in thresholds:
    # create scatter plot
    plt.scatter(recall, precision)

    # plot point for threshold score
    threshold_score = threshold
    index = thresholds.index(threshold_score)
    plt.scatter(recall_list[index], precision_list[index], color='red')

    # add text for threshold score above the point
    plt.text(recall_list[index] + 0.005, precision_list[index] - 0.001, str(threshold_score))

# Plot precision-recall curve
plt.plot(recall_list, precision_list)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid()
plt.legend(['Model Thresholds'], loc='upper right', bbox_to_anchor=(1, 1), markerscale=2, labelcolor='black', handlelength=0.5, handletextpad=0.5)
plt.gca().get_legend().legend_handles[0].set_color('red')
plt.savefig('Precision-recall curve.png')
plt.show()
