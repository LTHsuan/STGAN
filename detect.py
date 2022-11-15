from __future__ import division

from models.yolo_models import *

import os
import sys
import time
import argparse
import numpy as np

from PIL import Image

import torch
from torchvision import transforms
import cv2

class Face_detect:
    def __init__(self,age = {"cuda":True , "class_path":"configs/face.names" , "model_def" : "configs/yolov3-tiny-face.cfg" , "weights_path" : "experiments\stgan\checkpoints/yolov3-tiny-face_55000.weights" , "img_size" : 416
                             ,"conf_thres":0.8,"nms_thres":0.4}):
        self.device = torch.device("cuda" if torch.cuda.is_available()and age["cuda"] else "cpu")
        self.classes = self.load_classes(age["class_path"])  # Extracts class labels from file
        self.model_def = age["model_def"]
        self.weights_path = age["weights_path"]
        self.conf_thres = age["conf_thres"]
        self.nms_thres = age["nms_thres"]
        self.img_size = age["img_size"]
        print("divice : ", self.device)
        self.model = None
        self.init_model()
        self.Face = None

    def cv2_to_PLT(self,image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def load_classes(self,path):
        """
        Loads class labels at 'path'
        """
        fp = open(path, "r")
        names = fp.read().split("\n")[:-1]
        return names

    def PLT_to_cv2(self,image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def init_model(self):
        self.model = Darknet(self.model_def, img_size=self.img_size).to(self.device)
        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weights_path))

    def detect(self,image):
        self.model.eval()
        imgs = self.cv2_to_PLT(image)#

        loader = transforms.Compose([transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor()
            ])

        imgs = loader(imgs).to(self.device)
        imgs = torch.reshape(imgs, (1, imgs.shape[0], imgs.shape[1], imgs.shape[2]))

        with torch.no_grad():
            detections = self.model(imgs)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

        return detections

    def cut_image(self,image,point):
        w = (point[1][0]-point[0][0])*0.3
        h = (point[1][1]-point[0][1])*0.2

        x1 = point[0][0] - w if (point[0][0] - w) > 0 else 0
        y1 = point[0][1] - h if (point[0][1] - h) > 0 else 0
        x2 = point[1][0] + w if (point[1][0] + w) < image.shape[1] else image.shape[1]
        y2 = point[1][1] + h*0.5 if (point[1][1] + h) < image.shape[0] else image.shape[0]

        return self.cv2_to_PLT(image[int(y1) : int(y2) , int(x1): int(x2)])

    def rescale_bounding_boxs(self,img_detections,org_size,re_size):
        x1 = img_detections[:,0] * re_size[1] / org_size[1]
        x2 = img_detections[:,2] * re_size[1] / org_size[1]
        y1 = img_detections[:,1] * re_size[0] / org_size[0]
        y2 = img_detections[:,3] * re_size[0] / org_size[0]
        img_detections[:, 0] = x1
        img_detections[:, 2] = x2
        img_detections[:, 1] = y1
        img_detections[:, 3] = y2
        return img_detections

    def draw_result(self,image,img_detections):
        #image = self.PLT_to_cv2(image)#to cv2
        img = image.copy()
        if img_detections[0] is not None:
            max_face = [None,None,0]
            # Rescale boxes to original image
            image_shape = image.shape
            detections = self.rescale_bounding_boxs(img_detections[0], (self.img_size,self.img_size), [image_shape[0],image_shape[1]])
            #detections = img_detections[0]
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))
                # print(x1, y1, x2, y2)
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w*box_h >= max_face[2]:
                    max_face = [(x1, y1), (x2, y2),box_w*box_h]

                # Create a Rectangle
                cv2.rectangle(img,(x1, y1), (x2, y2), (0, 255, 0), 2)

            if max_face[2] != 0:
                self.Face = self.cut_image(image, [(x1, y1), (x2, y2)])
        #img = self.cv2_to_PLT(img)
        return self.Face, cv2.flip(img,1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny-face.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3-tiny-face_55000.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/face.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument('--cuda' , action='store_true', default=True,help='use gpu')
    opt = parser.parse_args()
    opt = vars(opt)
    print(opt)
    face_detect = Face_detect()

    cap = cv2.VideoCapture()
    cap.open(0)

    while (True):
        # 從攝影機擷取一張影像
        ret, frame = cap.read()
        image = face_detect.cv2_to_PLT(frame)
        img_detections = face_detect.detect(image)
        img = face_detect.draw_result(image, img_detections)
        img = face_detect.PLT_to_cv2(img)
        # 顯示圖片
        cv2.imshow('frame', img)
        if face_detect.Face  is not None:
            try:
                cv2.imshow('Face', face_detect.PLT_to_cv2(face_detect.Face))
            except:
                print("error")
        # 若按下 q 鍵則離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('q') or frame is None:
            break

    # 釋放攝影機
    cap.release()
    # 關閉所有 OpenCV 視窗
    #cv2.destroyAllWindows()


