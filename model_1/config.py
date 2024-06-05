import os

home = os.path.expanduser("~")
root_dirs = {
    'bird': home + '/temp/fine-grained-image_classification/model_1/Data/bird',
    'car':  home + '/temp/fine-grained-image_classification/model_1/Data/car',
    'air':  home + '/temp/fine-grained-image_classification/model_2/datasets/FGVC-aircraft',
    'dog':  home + '/temp/fine-grained-image_classification/model_1/Data/dog',
    'competition':  home + '/temp/fine-grained-image_classification/model_2/datasets/CompetitionData',

}

eval_test = False
class_nums = {
    'bird': 200,
    'car': 196,
    'air': 100,
    'dog': 120,
    'competition':100
}

HyperParams = {
    'alpha': 0.5,
    'beta':  0.5,
    'gamma': 1,
    'kind': 'competition',
    'bs': 20,
    'epoch': 200,
    'arch': 'resnet50'
}