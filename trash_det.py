import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import serial
import time

from imageai.Detection import ObjectDetection
import cv2




yolo_path = 'tiny-yolov3.pt'
model_path = 'vgg16_aug_model.pt'
model = models.vgg16_bn(pretrained=True)

classes=["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(classes))]) # Add our layer with 6 outputs
model.classifier = nn.Sequential(*features) # Replace the model classifier

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(yolo_path)
detector.loadModel()


test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

def image_loader_v2(image):
    color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(color_coverted)
    image = test_transforms(image).float()
    image = image.unsqueeze(0)
    return image

cam = cv2.VideoCapture(0) #0=front-cam, 1=back-cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

arduino = serial.Serial('/dev/cu.usbmodem14101', 9600, timeout=.1)
time.sleep(2)

while True:
    ## read frames
    ret, img = cam.read()
    ## predict yolo
    image, preds = detector.detectObjectsFromImage(input_image=img, 
                      custom_objects=None,
                      output_type="array",
                      minimum_percentage_probability=5,
                      display_percentage_probability=False,
                      display_object_name=True)
    ## display predictions
    cv2.imshow("", img)
    if len(preds) > 0:
      image = image_loader_v2(img)
      model.eval()
      output = model(image)
      output = nn.ReLU()(output)
      output = nn.LogSoftmax(dim=1)(output)
      output[:, 5] += 3
      _, pred = torch.max(output, 1)
      print(classes[pred])

      arduino.write(str(pred.item()).encode())
      time.sleep(2)

      line = arduino.readline()
      print(line)

    ## press q or Esc to quit    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
## close camera
cam.release()
cv2.destroyAllWindows()
arduino.close()


