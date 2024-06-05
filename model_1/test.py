from __future__ import print_function
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
import torch.nn.functional as F

import os
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb

from torch.utils.data.dataloader import DataLoader
import numpy as np
from model import FBSD  # Ensure your model is imported correctly
from datesets import get_trainAndtest
from config import class_nums, HyperParams, eval_test


from http.client import responses
import requests
import json


def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
        



def test(current=False):
    
    # Output dir
    output_dir = HyperParams['kind'] + '_' + HyperParams['arch'] + '_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    trainset, testset = get_trainAndtest()
    testloader = DataLoader(testset, shuffle=False, num_workers=8)

    
    net = FBSD(class_num=class_nums[HyperParams['kind']], arch=HyperParams['arch'])
    net = net.cuda()


    model_type = 'best_model'
    if current:
        model_type = 'current_model'
    # Load pretrained model if it exists
    model_path = f'./{output_dir}/{model_type}.pth'
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)

        # Load the pretrained weights into the model
        net.load_state_dict(checkpoint, strict=False)

        # Find the correct layer to extract the number of input features
        for layer in reversed(net.classifier_concat):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                break
        final_layer = nn.Linear(in_features, class_nums[HyperParams['kind']])  # Create a new final layer
        final_layer = final_layer.cuda()
        net.classifier_concat[4] = final_layer  # Replace the final layer in your model

                
        print(f'Loaded pretrained model from {model_path}')
    else:
        print('No pretrained model found, starting from scratch.')
    
    net.eval()
    correct_com = 0
    total = 0
    predicted_results = []
    img_paths = []
    for batch_idx, (inputs,img_path)  in enumerate(testloader):
        img_paths.append(img_path)
        inputs = inputs.cuda()
        with torch.no_grad():
            output_1, output_2, output_3, output_concat = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

        _, predicted_com = torch.max(outputs_com.data, 1)
        predicted_results.append(predicted_com.cpu().numpy())

    
    train_data = os.listdir('../model_2/datasets/CompetitionData/train')
    train_data.sort()

    class_dict = {}
    i = 0
    for item in train_data:
        class_dict[str(i)] = item.split('_')[0]    
        i+=1 
    preds  = {}
    for idx, result in enumerate(predicted_results):
        preds[img_paths[idx][0].split('/')[-1]] = class_dict[str(result[0])]
    
    res = {
    "images": preds,
    "groupname": "duck_duck_goose"
    }

    submit(res)
    
    return predicted_results, img_paths

