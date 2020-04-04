from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import math

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
                "Location to read the metadata related to the training (generated when training).",
                default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

#if not options.test_path:   # if filename is not given
#    parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

#if C.network == 'resnet50':
#    import keras_frcnn.resnet as nn
#elif C.network == 'vgg':
import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

#img_path = "/home/khalid/Documents/Thesis_work/Experiments_work/Cats_dogs_classifier/test_images" #options.test_path


img_path = "./test_images"
#img_path = "./sample"


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape
        
    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio    

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

#if C.network == 'resnet50':
#    num_features = 1024
#elif C.network == 'vgg':
num_features = 512

#if K.image_dim_ordering() == 'th':
#    input_shape_img = (3, None, None)
#    input_shape_features = (num_features, None, None)
#else:
input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

#from keras_frcnn.simple_parser import get_data

#all_imgs, _, _ = get_data("annotation_test.txt")
#
#all_imgs[0]


visualise = True

def plot_gnd_tth_ellipses(image, gnd_tth_annotations):
    
    for i in range(len(gnd_tth_annotations["bboxes"])):
                
        x, y, x_axis, y_axis = gnd_tth_annotations["bboxes"][i]["x"], gnd_tth_annotations["bboxes"][i]["y"], gnd_tth_annotations["bboxes"][i]["x_axis"], \
                                gnd_tth_annotations["bboxes"][i]["y_axis"]
        
        cv2.ellipse(image, (x, y), (x_axis, y_axis), 0,0,360,(0,0,255),2)
        
        print("Drew an ellipse")
        
    return image

def convert_to_rect(dsa):
#    dsa = np.array(bboxes["RBC"])
    
    dsa[:,0] = dsa[:,0] - dsa[:,2]
    dsa[:,1] = dsa[:,1] - dsa[:,3]
    dsa[:,2] = dsa[:,0] + (2 * dsa[:,2])
    dsa[:,3] = dsa[:,1] + (2 * dsa[:,3])

    
    return dsa

def convert_to_ellip(x,y,w,h):

    x_axis = (w - x) / 2
    y_axis = (h - y) / 2
    x = (x + w) / 2
    y = (y + h) / 2
    
    return int(x),int(y),int(x_axis),int(y_axis)

## tw and th are adjusted w.r.t the rectangles, so need to convert it to ellipse
def bbox_resize(x, y, w, h, tx, ty, tw, th):
    try:
        
        
        
        cx = x + (w/2)
        cy = y + (h/2)
         
#        cx=x
#        cy=y
        
        cx1 = (tx) * (w/2) + cx
        cy1 = (ty) * (h/2) + cy
        
        w1 = math.exp(tw) * (w/2)
        h1 = math.exp(th) * (h/2)
        
        x1 = cx1# - ((w1/2))
        y1 = cy1# - ((h1/2))
        
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h            
    ##
#iou(rectangles=False)

def apply_regr(x, y, w, h, tx, ty, tw, th):
	try:
		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		w1 = math.exp(tw) * w
		h1 = math.exp(th) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.
		x1 = int(round(x1))
		y1 = int(round(y1))
		w1 = int(round(w1))
		h1 = int(round(h1))

		return x1, y1, w1, h1

	except ValueError:
		return x, y, w, h
	except OverflowError:
		return x, y, w, h
	except Exception as e:
		print(e)
		return x, y, w, h
    

import pickle
##Load validation data from the pickle file
with open("validation_animals.pkl", "rb") as f:
    validation_data = pickle.load(f)

##        
#for idx, img_name in enumerate(sorted(os.listdir(img_path))):
for idx, img_name in enumerate(validation_data):
    
    if idx > 200:
        break
#    idx=0
#    img_name="Abyssinian_48.jpg"
    

#    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
#        continue

    print(img_name)
    st = time.time()
#    filepath = os.path.join(img_path,img_name)

    filepath = "/home/khalid/Documents/Thesis_work/Experiments_work/Oxford_dataset/images/"+img_name
    img = cv2.imread(filepath)
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
    
    
    X, ratio = format_img(img, C)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    # Y2 will output coordinates of ellipses
    [Y1, Y2, F] = model_rpn.predict(X)
    

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    rectangles = False
    
    bboxes = {}
    probs = {}


    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)    
#         jk=0
        if jk == R.shape[0]//C.num_rois:
        #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded
            
            
        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
            
            if cls_name not in bboxes:
                
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            print(ii,cls_name)
            
            if not rectangles:
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = bbox_resize(x, y, w, h, tx, ty, tw, th)
#                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
#                    print("Exception occured here")
                    pass
            else:
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
#                    x, y, w, h = bbox_resize(x, y, w, h, tx, ty, tw, th)
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                    print(ii,jk,cls_name)
                except:
#                    print("Exception occured here")
                    pass

            if rectangles:
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            else:
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(w), C.rpn_stride*(h)])
                
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
    
    if rectangles:
        plot_rectangles()

    else:
#        img = cv2.imread(filepath)
#        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
    
        for key in bboxes:
            
            if key == "bg":
                continue
            
            bbox = np.array(bboxes[key])
            
            bbox = convert_to_rect(bbox)
            
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.1)
    
            for jk in range(1):#range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
    
                (x, y, w, h) = get_real_coordinates(ratio, x1, y1, x2, y2)
                
                
                
    #        for i in range(len(bboxes["RBC"])):
    #    
    #            x,y,x_axis,y_axis = bboxes["RBC"][i]                
        
    #            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.3)            
        
    #            if x<0 or y<0:
    #                continue
                
                (x,y,x_axis,y_axis) = convert_to_ellip(x,y,w,h)
                print((x,y,x_axis,y_axis))
                cv2.ellipse(img, (x, y), (x_axis, y_axis), 0,0,360,(0,255,0),2)
                
                textLabel = '{}: {}%'.format(key,int(100*new_probs[jk]))

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)

                textOrg = (int(x+30), int(y-y_axis))
                
                
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
                print("Here")        
            
            
        
#        img = plot_gnd_tth_ellipses(img, all_imgs[idx])
    
#        cv2.imwrite("/home/khalid/Documents/Thesis_work/Experiments_work/Cats_dogs_classifier/Output/"+img_name, img)
                
        cv2.imwrite("./Output_1/"+img_name, img)
#        cv2.imshow('img', img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()


    # convert from (x1,y1,x2,y2) to (x,y,w,h)
#    R[:, 2] -= R[:, 0]
#    R[:, 3] -= R[:, 1]
#
#    # apply the spatial pyramid pooling to the proposed regions
#    bboxes = {}
#    probs = {}
#
#    for jk in range(R.shape[0]//C.num_rois + 1):
#        
#        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
#        if ROIs.shape[1] == 0:
#            break
#
#        if jk == R.shape[0]//C.num_rois:
#            #pad R
#            curr_shape = ROIs.shape
#            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
#            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
#            ROIs_padded[:, :curr_shape[1], :] = ROIs
#            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
#            ROIs = ROIs_padded
#
#        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
#
#        for ii in range(P_cls.shape[1]):
#                                                       ## Checking if it is background or not     
#            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
##                print("Nothing found")
#                continue
#
#            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
#
#            if cls_name not in bboxes:
#                bboxes[cls_name] = []
#                probs[cls_name] = []
#
#            (x, y, w, h) = ROIs[0, ii, :]
#
#            cls_num = np.argmax(P_cls[0, ii, :])
#            try:
#                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
#                tx /= C.classifier_regr_std[0]
#                ty /= C.classifier_regr_std[1]
#                tw /= C.classifier_regr_std[2]
#                th /= C.classifier_regr_std[3]
#                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
#            except:
#                print("Exception occured here")
#                pass
#            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
#            probs[cls_name].append(np.max(P_cls[0, ii, :]))

def iou(rectangles):
    
    bboxes = {}
    probs = {}


    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)    
#         jk=0
        if jk == R.shape[0]//C.num_rois:
        #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded
            
            
        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):
#            print(ii)
#            ii=18
#            print(np.argmax(P_cls[0, ii, :]))
            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
            
#            print(cls_name)
            if cls_name not in bboxes:
                
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
#            print(ii,cls_name)
            
            if not rectangles:
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = bbox_resize(x, y, w, h, tx, ty, tw, th)
#                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
#                    print("Exception occured here")
                    pass
            else:
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
#                    x, y, w, h = bbox_resize(x, y, w, h, tx, ty, tw, th)
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                    print(ii,jk,cls_name)
                except:
#                    print("Exception occured here")
                    pass

            if rectangles:
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            else:
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(w), C.rpn_stride*(h)])
                
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
    
    if rectangles:
        plot_rectangles()

    else:
        img = cv2.imread(filepath)
    
        for i in range(len(bboxes["WBC"])):
    
            x,y,x_axis,y_axis = bboxes["WBC"][i]            
            
            if x<0 or y<0:
                continue
            (x,y,x_axis,y_axis) = get_real_coordinates(ratio, x,y,x_axis,y_axis)
            
            cv2.ellipse(img, (x, y), (x_axis, y_axis), 0,0,360,(255,255,0),2)
            print("Here")        
            
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def plot_rectangles():            
    all_dets = []

    for key in bboxes:
        if key == "RBC" or key == "bg":
            continue
        print(key)
        bbox = np.array(bboxes[key])
        
        
        
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.3)

        

        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

#            cv2.ellipse(img, (real_x1, real_y1), (real_x2, real_y2), 0,0,(255,0,0),1)

            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

#        print('Elapsed time = {}'.format(time.time() - st))
    print(all_dets)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return
#    cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
