import torch
from tqdm import tqdm
import os
import numpy as np
from config import coordinates_cat, proposalN, set, vis_num
import warnings
warnings.filterwarnings('ignore')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

def eval(model, testloader, epoch):
    model.eval()
    print('Evaluating')

    predictions = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            images = data.cuda()
            local_logits, local_imgs = model(images, epoch, i, 'test', DEVICE)[-2:]

            # Get predictions from raw and local logits
            local_pred = local_logits.max(1, keepdim=True)[1]

            # Store predictions
            predictions.extend(local_pred.cpu().numpy())

    return predictions
