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

def test():
    
    # Output dir
    output_dir = HyperParams['kind'] + '_' + HyperParams['arch'] + '_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    trainset, testset = get_trainAndtest()
    testloader = DataLoader(testset, batch_size=HyperParams['bs'], shuffle=False, num_workers=8)

    ####################################################
    print("dataset: ", HyperParams['kind'])
    print("backbone: ", HyperParams['arch'])
    print("trainset: ", len(trainset))
    print("testset: ", len(testset))
    print("classnum: ", class_nums[HyperParams['kind']])
    ####################################################

    net = FBSD(class_num=class_nums[HyperParams['kind']], arch=HyperParams['arch'])
    net = net.cuda()




    # Load pretrained model if it exists
    model_path = f'./{output_dir}/best_model.pth'
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)

        # checkpoint.pop('classifier_concat.4.weight', None)
        # checkpoint.pop('classifier_concat.4.bias', None)
        # checkpoint.pop('classifier1.4.weight', None)
        # checkpoint.pop('classifier1.4.bias', None)
        # checkpoint.pop('classifier2.4.weight', None)
        # checkpoint.pop('classifier2.4.bias', None)
        # checkpoint.pop('classifier3.4.weight', None)
        # checkpoint.pop('classifier3.4.bias', None)

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
    for batch_idx, inputs in enumerate(testloader):
        inputs = inputs.cuda()
        with torch.no_grad():
            output_1, output_2, output_3, output_concat = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

        _, predicted_com = torch.max(outputs_com.data, 1)
        predicted_results.append(predicted_com.cpu().numpy())

    return predicted_results 

result = test()
print(result)