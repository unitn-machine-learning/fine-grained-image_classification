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
from config import class_nums
from config import HyperParams

def train():
    # Initialize wandb
    wandb.init(project="my_project", config=HyperParams)
    config = wandb.config

    # Output dir
    output_dir = config.kind + '_' + config.arch + '_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    trainset, testset = get_trainAndtest()
    trainloader = DataLoader(trainset, batch_size=config.bs, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size=config.bs, shuffle=False, num_workers=8)

    ####################################################
    print("dataset: ", config.kind)
    print("backbone: ", config.arch)
    print("trainset: ", len(trainset))
    print("testset: ", len(testset))
    print("classnum: ", class_nums[config.kind])
    ####################################################

    net = FBSD(class_num=class_nums[config.kind], arch=config.arch)
    net = net.cuda()
    netp = nn.DataParallel(net).cuda()

    CELoss = nn.CrossEntropyLoss()

    ########################
    new_params, old_params = net.get_params()
    new_layers_optimizer = optim.SGD(new_params, momentum=0.9, weight_decay=5e-4, lr=0.002)
    old_layers_optimizer = optim.SGD(old_params, momentum=0.9, weight_decay=5e-4, lr=0.0002)
    new_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_layers_optimizer, config.epoch, 0)
    old_layers_optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(old_layers_optimizer, config.epoch, 0)

    scaler = torch.cuda.amp.GradScaler()

    # Load pretrained model if it exists
    start_epoch = 0
    best_val_acc = 0
    model_path = f'./{output_dir}/best_model.pth'
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)

        checkpoint.pop('classifier_concat.4.weight', None)
        checkpoint.pop('classifier_concat.4.bias', None)
        checkpoint.pop('classifier1.4.weight', None)
        checkpoint.pop('classifier1.4.bias', None)
        checkpoint.pop('classifier2.4.weight', None)
        checkpoint.pop('classifier2.4.bias', None)
        checkpoint.pop('classifier3.4.weight', None)
        checkpoint.pop('classifier3.4.bias', None)

        # Load the pretrained weights into the model
        net.load_state_dict(checkpoint, strict=False)

        # Find the correct layer to extract the number of input features
        for layer in reversed(net.classifier_concat):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                break
        final_layer = nn.Linear(in_features, class_nums[config.kind])  # Create a new final layer
        final_layer = final_layer.cuda()
        net.classifier_concat[4] = final_layer  # Replace the final layer in your model

                
        print(f'Loaded pretrained model from {model_path}')
    else:
        print('No pretrained model found, starting from scratch.')

    for epoch in range(start_epoch, config.epoch):
        print('\nEpoch: %d' % epoch)
        start_time = datetime.now()
        print("start time: ", start_time.strftime('%Y-%m-%d-%H:%M:%S'))
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            inputs, targets = inputs.cuda(), targets.cuda()

            with torch.cuda.amp.autocast():
                output_1, output_2, output_3, output_concat = netp(inputs)

                # Adjust optimizer lr
                new_layers_optimizer_scheduler.step()
                old_layers_optimizer_scheduler.step()

                # Overall update
                loss1 = CELoss(output_1, targets) * 2
                loss2 = CELoss(output_2, targets) * 2
                loss3 = CELoss(output_3, targets) * 2
                concat_loss = CELoss(output_concat, targets)
                loss = loss1 + loss2 + loss3 + concat_loss

            new_layers_optimizer.zero_grad()
            old_layers_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(new_layers_optimizer)
            scaler.step(old_layers_optimizer)
            scaler.update()

            # Training log
            _, predicted = torch.max((output_1 + output_2 + output_3 + output_concat).data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 50 == 0:
                print('Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))
        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        
        # Log to wandb
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "epoch": epoch})

        # Evaluate
        val_acc = test(net, testloader)
        torch.save(net.state_dict(), f'./{output_dir}/current_model.pth')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), f'./{output_dir}/best_model.pth')
        print("best result: ", best_val_acc)
        print("current result: ", val_acc)
        end_time = datetime.now()
        print("end time: ", end_time.strftime('%Y-%m-%d-%H:%M:%S'))

        # Log validation accuracy to wandb
        wandb.log({"val_acc": val_acc, "best_val_acc": best_val_acc, "epoch": epoch})

def test(net, testloader):
    net.eval()
    correct_com = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            output_1, output_2, output_3, output_concat = net(inputs)
            outputs_com = output_1 + output_2 + output_3 + output_concat

        _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        correct_com += predicted_com.eq(targets.data).cpu().sum()
    test_acc_com = 100. * float(correct_com) / total

    return test_acc_com 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    set_seed(666)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()
