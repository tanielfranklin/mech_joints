# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
import os
import glob as glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import pathlib

class_names = ['joint']
colors = np.random.uniform(0, 255, size=(len(class_names), 3))
np.random.seed(42)

def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize the coordinates
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)
        width = xmax - xmin
        height = ymax - ymin
        
        class_name = class_names[int(labels[box_num])]
        
        cv2.rectangle(
            image, 
            (xmin, ymin), (xmax, ymax),
            color=colors[class_names.index(class_name)],
            thickness=2
        ) 

        font_scale = min(1,max(3,int(w/500)))
        font_thickness = min(2, max(10,int(w/50)))
        
        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        # Text width and height
        tw, th = cv2.getTextSize(
            class_name, 
            0, fontScale=font_scale, thickness=font_thickness
        )[0]
        p2 = p1[0] + tw, p1[1] + -th - 10
        cv2.rectangle(
            image, 
            p1, p2,
            color=colors[class_names.index(class_name)],
            thickness=-1,
        )
        cv2.putText(
            image, 
            class_name,
            (xmin+1, ymin-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    return image

def discard_pic_without_label(image_paths,label_paths):
    #paths of images
    im=glob.glob(image_paths)
    im.sort()
    im_names=[pathlib.Path(i).stem for i in im]
    #paths of labels
    lb=glob.glob(label_paths)
    lb.sort()
    lb_names=[pathlib.Path(i).stem for i in lb]
    im_new,lb_new=[],[]
    #check if picture has a label
    for ind,name in enumerate(im_names):
        if name in lb_names: 
            index=lb_names.index(name)
            im_new.append(im[ind])
            lb_new.append(lb[index])
    #return only pictures with label
    return [im_new,lb_new]

# Function to plot images with the bounding boxes.
def plot(image_label, num_samples):
    
    all_training_images = image_label[0]
    all_training_labels = image_label[1]

    
    num_images = len(all_training_images)
    
    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()
    
    
plot(discard_pic_without_label('./train_data/images/train/*',
                                './train_data/labels/train/*'),
    num_samples=4
)

