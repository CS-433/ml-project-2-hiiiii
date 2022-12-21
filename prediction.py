import torch
import torchvision
import cv2
import os
import numpy as np
import constants as cst
from data.transforms import *

################################################################################
# PREDICTION
################################################################################

def split_test_image(image):
    '''Split a 608x608 image into 4 400x400 images'''
    image_parts = []
    for i in [0, 208]:
        for j in [0, 208]:
            image_parts.append(image[i : i + 400, j : j + 400])
    return image_parts

def predict_image_part(image_part, model, device):
    '''Predict all transformed versions of a 400x400 image part and return the average prediction'''
    all_predictions_part = np.zeros((400, 400))
    for transform, inverse_transform in zip(
        get_test_transforms(), get_inverse_test_transforms()
    ):
        image_part_transformed = transform(image=image_part)["image"]
        image_part_transformed = image_part_transformed.unsqueeze(0).to(device)
        prediction = model(image_part_transformed)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
        prediction = prediction.cpu().numpy()
        prediction = prediction[0, 0, :, :]
        prediction_transformed = inverse_transform(image=prediction)["image"]
        prediction_transformed = prediction_transformed[0, :, :].numpy()
        all_predictions_part += prediction_transformed
    # treshold the prediction
    all_predictions_part[all_predictions_part < len(get_test_transforms()) / 2] = 0
    all_predictions_part[all_predictions_part >= len(get_test_transforms()) / 2] = 1
    return all_predictions_part
    
def combine_image_parts_predictions(predictions):
    '''Combine the 4 predictions into a 608x608 image'''
    prediction = np.zeros((608, 608))
    prediction[0:400, 0:400] += predictions[0]
    prediction[0:400, 208:608] += predictions[1]
    prediction[208:608, 0:400] += predictions[2]
    prediction[208:608, 208:608] += predictions[3]
    # correct the overlapping parts
    prediction[208:400, 0:208][prediction[208:400, 0:208] >= 1] = 1
    prediction[0:208, 208:400][prediction[0:208, 208:400] >= 1] = 1
    prediction[400:608, 208:400][prediction[400:608, 208:400] >= 1] = 1
    prediction[208:400, 400:608][prediction[208:400, 400:608] >= 1] = 1
    prediction[208:400, 208:400][prediction[208:400, 208:400] < 2] = 0
    prediction[208:400, 208:400][prediction[208:400, 208:400] >= 2] = 1
    return prediction

def predict_image(image, image_folder, model, device):
    '''Predict on a single image'''
    model.eval()
    all_predictions = []
    # split the image into 4 parts
    for image_part in split_test_image(image):
        # predict on the 4 parts
        prediction_part = predict_image_part(image_part, model, device)
        all_predictions.append(prediction_part)
    # combine the 4 predictions
    prediction = combine_image_parts_predictions(all_predictions)
    # save the image
    torchvision.utils.save_image(
        torch.tensor(prediction), cst.TEST_IMAGE_DIR + image_folder + "/" + image_folder + "_pred.png"
    )

def predict_test_images(model):
    '''Predict on all test images'''
    for image_folder in os.listdir(cst.TEST_IMAGE_DIR):
        img = cv2.imread(cst.TEST_IMAGE_DIR + image_folder + "/" + image_folder + ".png")
        img = np.array(img)
        predict_image(img, image_folder, model, cst.DEVICE)
