import os

home = os.path.expanduser("~")
root_dirs = {
    'bird': home + '/temp/fine-grained-image_classification/model_1/Data/bird',
    'car':  home + '/temp/fine-grained-image_classification/model_1/Data/car',
    'air':  home + '/temp/fine-grained-image_classification/model_1/Data/aircraft',
    'dog':  home + '/temp/fine-grained-image_classification/model_1/Data/dog'
}

class_nums = {
    'bird': 200,
    'car': 196,
    'air': 100,
    'dog': 120
}

HyperParams = {
    'alpha': 0.5,
    'beta':  0.5,
    'gamma': 1,
    'kind': 'bird',
    'bs': 20,
    'epoch': 200,
    'arch': 'resnet50'
}
