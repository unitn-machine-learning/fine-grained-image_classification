from utils.indices2coordinates import indices2coordinates
from utils.compute_window_nums import compute_window_nums
import numpy as np

CUDA_VISIBLE_DEVICES = '0'  # The current version only supports one GPU training


set = 'CUB'  # Different dataset with different
model_name = 'ML-Project_FGIA-Model_2'

batch_size = 6
vis_num = batch_size  # The number of visualized images in tensorboard
eval_trainset = False  # Whether or not evaluate trainset
save_interval = 5
max_checkpoint_num = 200
end_epoch = 10
init_lr = 0.001
lr_milestones = [60, 100]
lr_decay_rate = 0.1
weight_decay = 1e-4
stride = 32
channels = 2048
input_size = 448

# The pth path of pretrained model
pretrain_path = './models/pretrained/resnet50-19c8e357.pth'

dataset_paths = {'Aircraft': './datasets/FGVC-aircraft', 
                 'CUB': './datasets/CUB_200_2011',
                 'CAR':'./datasets/Stanford_Cars',
                 'Competition':'./datasets/CompetitionData',
                 }

model_path = f'./checkpoint/{set}'  # pth save path
root =   dataset_paths[set]# dataset path

if set == 'CUB':
    num_classes = 200
    # windows info for CUB
    N_list = [2, 3, 2]
    proposalN = sum(N_list)  # proposal window num
    window_side = [128, 192, 256]
    iou_threshs = [0.25, 0.25, 0.25]
    ratios = [[4, 4], [3, 5], [5, 3],
              [6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]
else:
    # windows info for CAR and Aircraft
    N_list = [3, 2, 1]
    proposalN = sum(N_list)  # proposal window num
    window_side = [192, 256, 320]
    iou_threshs = [0.25, 0.25, 0.25]
    ratios = [[6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
              [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]
    if set == 'CAR':
        num_classes = 196
    elif set == 'Aircraft':
        num_classes = 100


'''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]

#debug statement
for i, indices_ndarray in enumerate(indices_ndarrays):
    print(f"Calling indices2coordinates with indices_ndarray: i: {i},{indices_ndarray}, stride: {stride}, input_size: {input_size}, ratio: {ratios[i]}")
    coordinates = indices2coordinates(indices_ndarray, stride, input_size, ratios[i])

coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # Coordinates of the bounding boxes
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
if set == 'CUB':
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:6]), sum(window_nums[6:])]
else:
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:8]), sum(window_nums[8:])]
