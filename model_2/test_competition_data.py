#coding=utf-8
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from config import input_size, root, proposalN, channels, num_classes, set
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet
import warnings
warnings.filterwarnings('ignore')

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

from http.client import responses
import requests
import json
from datasets.dataset import CompetitionDataset
from torch.utils.data import DataLoader



def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
        
def create_testloader(data_dir, input_size, batch_size):
    test_dataset = CompetitionDataset(input_size=input_size, root=data_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def test(root = './datasets/CompetitionData',labels_path='datasets/CompetitionData/labels.txt', pth_path = './models/competition.pth',num_classes = num_classes):
    
    label_txt_file = open(labels_path)
    class_dict = {}
    for line in label_txt_file:
        class_dict[line.split(' ')[0]] = line.split(' ')[1][:-1]

    #load dataset
    testloader = create_testloader(root, input_size, batch_size = 1)

    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if os.path.exists(pth_path):
        epoch = auto_load_resume(model, pth_path, status='test')
    else:
        sys.exit('The pth doesn\'t exist.')
    
    print('Testing')
    
    model.eval()
    preds = {}
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
                x, name = data
                x = x.to(DEVICE)
                local_logits, local_imgs = model(x, epoch, i, 'test', DEVICE)[-2:]
                # local
                pred = local_logits.max(1, keepdim=True)[1]
                preds[name[0].split('/')[-1]] =  '0' + class_dict[str(pred.cpu().numpy()[0][0])]

    res = {
    "images": preds,
    "groupname": "duck_duck_goose"
    }
    submit(res)
    return preds
