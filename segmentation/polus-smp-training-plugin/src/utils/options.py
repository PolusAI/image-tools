from typing import Dict
from typing import List
from typing import Set
from typing import Type

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.utils.base import Metric
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import Optimizer

__all__ = [
    'MODELS',
    'ENCODERS',
    'OPTIMIZERS',
    'LOSSES',
    'METRICS',
    'MODEL_NAMES',
    'BASE_ENCODERS',
    'ENCODER_VARIANTS',
    'ENCODER_WEIGHTS',
    'OPTIMIZER_NAMES',
    'LOSS_NAMES',
]

MODELS: Dict[str, Type[SegmentationModel]] = {
    'Unet': smp.Unet,
    'UnetPlusPlus': smp.UnetPlusPlus,
    'MAnet': smp.MAnet,
    'Linknet': smp.Linknet,
    'FPN': smp.FPN,
    'PSPNet': smp.PSPNet,
    'PAN': smp.PAN,
    'DeepLabV3': smp.DeepLabV3,
    'DeepLabV3Plus': smp.DeepLabV3Plus,
}

MODEL_NAMES: List[str] = list(MODELS.keys())

# A dictionary of base encoder names to a dict of specific encoder names.
# The inner dictionaries are encoder names to their pretrained weights
# { base-encoder: { encoder: [weights] } }
ENCODERS: Dict[str, Dict[str, List[str]]] = {
    'ResNet': {
        'resnet18': ['imagenet', 'ssl', 'swsl'],
        'resnet34': ['imagenet'],
        'resnet50': ['imagenet', 'ssl', 'swsl'],
        'resnet101': ['imagenet'],
        'resnet152': ['imagenet'],
    },
    'ResNeXt': {
        'resnext50_32x4d': ['imagenet', 'ssl', 'swsl'],
        'resnext101_32x4d': ['ssl', 'swsl'],
        'resnext101_32x8d': ['imagenet', 'instagram', 'ssl', 'swsl'],
        'resnext101_32x16d': ['instagram', 'ssl', 'swsl'],
        'resnext101_32x32d': ['instagram'],
        'resnext101_32x48d': ['instagram'],
    },
    'ResNeSt': {
        'timm-resnest14d': ['imagenet'],
        'timm-resnest26d': ['imagenet'],
        'timm-resnest50d': ['imagenet'],
        'timm-resnest101e': ['imagenet'],
        'timm-resnest200e': ['imagenet'],
        'timm-resnest269e': ['imagenet'],
        'timm-resnest50d_4s2x40d': ['imagenet'],
        'timm-resnest50d_1s4x24d': ['imagenet'],
    },
    'Res2Ne(X)t': {
        'timm-res2net50_26w_4s': ['imagenet'],
        'timm-res2net101_26w_4s': ['imagenet'],
        'timm-res2net50_26w_6s': ['imagenet'],
        'timm-res2net50_26w_8s': ['imagenet'],
        'timm-res2net50_48w_2s': ['imagenet'],
        'timm-res2net50_14w_8s': ['imagenet'],
        'timm-res2next50': ['imagenet'],
    },
    'RegNet(x/y)': {
        'timm-regnetx_002': ['imagenet'],
        'timm-regnetx_004': ['imagenet'],
        'timm-regnetx_006': ['imagenet'],
        'timm-regnetx_008': ['imagenet'],
        'timm-regnetx_016': ['imagenet'],
        'timm-regnetx_032': ['imagenet'],
        'timm-regnetx_040': ['imagenet'],
        'timm-regnetx_064': ['imagenet'],
        'timm-regnetx_080': ['imagenet'],
        'timm-regnetx_120': ['imagenet'],
        'timm-regnetx_160': ['imagenet'],
        'timm-regnetx_320': ['imagenet'],
        'timm-regnety_002': ['imagenet'],
        'timm-regnety_004': ['imagenet'],
        'timm-regnety_006': ['imagenet'],
        'timm-regnety_008': ['imagenet'],
        'timm-regnety_016': ['imagenet'],
        'timm-regnety_032': ['imagenet'],
        'timm-regnety_040': ['imagenet'],
        'timm-regnety_064': ['imagenet'],
        'timm-regnety_080': ['imagenet'],
        'timm-regnety_120': ['imagenet'],
        'timm-regnety_160': ['imagenet'],
        'timm-regnety_320': ['imagenet'],
    },
    'GERNet': {
        'timm-gernet_s': ['imagenet'],
        'timm-gernet_m': ['imagenet'],
        'timm-gernet_l': ['imagenet'],
    },
    'SE-Net': {
        'senet154': ['imagenet'],
        'se_resnet50': ['imagenet'],
        'se_resnet101': ['imagenet'],
        'se_resnet152': ['imagenet'],
        'se_resnext50_32x4d': ['imagenet'],
        'se_resnext101_32x4d': ['imagenet'],
    },
    'SK-ResNe(X)t': {
        'timm-skresnet18': ['imagenet'],
        'timm-skresnet34': ['imagenet'],
        'timm-skresnext50_32x4d': ['imagenet'],
    },
    'DenseNet': {
        'densenet121': ['imagenet'],
        'densenet169': ['imagenet'],
        'densenet201': ['imagenet'],
        'densenet161': ['imagenet'],
    },
    'Inception': {
        'inceptionresnetv2': ['imagenet', 'imagenet+background'],
        'inceptionv4': ['imagenet', 'imagenet+background'],
        'xception': ['imagenet'],
    },
    'EfficientNet': {
        'efficientnet-b0': ['imagenet'],
        'efficientnet-b1': ['imagenet'],
        'efficientnet-b2': ['imagenet'],
        'efficientnet-b3': ['imagenet'],
        'efficientnet-b4': ['imagenet'],
        'efficientnet-b5': ['imagenet'],
        'efficientnet-b6': ['imagenet'],
        'efficientnet-b7': ['imagenet'],
        'timm-efficientnet-b0': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b1': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b2': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b3': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b4': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b5': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b6': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b7': ['imagenet', 'advprop', 'noisy-student'],
        'timm-efficientnet-b8': ['imagenet', 'advprop'],
        'timm-efficientnet-l2': ['noisy-student'],
        'timm-efficientnet-lite0': ['imagenet'],
        'timm-efficientnet-lite1': ['imagenet'],
        'timm-efficientnet-lite2': ['imagenet'],
        'timm-efficientnet-lite3': ['imagenet'],
        'timm-efficientnet-lite4': ['imagenet'],
    },
    'MobileNet': {
        'mobilenet_v2': ['imagenet'],
        'timm-mobilenetv3_large_075': ['imagenet'],
        'timm-mobilenetv3_large_100': ['imagenet'],
        'timm-mobilenetv3_large_minimal_100': ['imagenet'],
        'timm-mobilenetv3_small_075': ['imagenet'],
        'timm-mobilenetv3_small_100': ['imagenet'],
        'timm-mobilenetv3_small_minimal_100': ['imagenet'],
    },
    'DPN': {
        'dpn68': ['imagenet'],
        'dpn68b': ['imagenet+5k'],
        'dpn92': ['imagenet+5k'],
        'dpn98': ['imagenet'],
        'dpn107': ['imagenet+5k'],
        'dpn131': ['imagenet'],
    },
    'VGG': {
        'vgg11': ['imagenet'],
        'vgg11_bn': ['imagenet'],
        'vgg13': ['imagenet'],
        'vgg13_bn': ['imagenet'],
        'vgg16': ['imagenet'],
        'vgg16_bn': ['imagenet'],
        'vgg19': ['imagenet'],
        'vgg19_bn': ['imagenet'],
    },
}

BASE_ENCODERS: List[str] = list(ENCODERS.keys())
ENCODER_VARIANTS: List[str] = list()
ENCODER_WEIGHTS: Set[str] = {'random'}

for encoder, variants in ENCODERS.items():
    ENCODER_VARIANTS.extend(variants.keys())

    for variant, weights in variants.items():
        ENCODER_WEIGHTS.update(weights)

OPTIMIZERS: Dict[str, Type[Optimizer]] = {
    'Adadelta': torch.optim.Adadelta,
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'SparseAdam': torch.optim.SparseAdam,
    'Adamax': torch.optim.Adamax,
    'ASGD': torch.optim.ASGD,
    'LBFGS': torch.optim.LBFGS,
    'RMSprop': torch.optim.RMSprop,
    'Rprop': torch.optim.Rprop,
    'SGD': torch.optim.SGD,
}

OPTIMIZER_NAMES: List[str] = list(OPTIMIZERS.keys())

LOSSES: Dict[str, Type[TorchLoss]] = {
    'JaccardLoss': smp.losses.JaccardLoss,
    'DiceLoss': smp.losses.DiceLoss,
    'TverskyLoss': smp.losses.TverskyLoss,
    'FocalLoss': smp.losses.FocalLoss,
    'LovaszLoss': smp.losses.LovaszLoss,
    'SoftBCEWithLogitsLoss': smp.losses.SoftBCEWithLogitsLoss,
    'SoftCrossEntropyLoss': smp.losses.SoftCrossEntropyLoss,
    'MCCLoss': smp.losses.MCCLoss,
}

LOSS_NAMES: List[str] = list(LOSSES.keys())

METRICS: Dict[str, Type[Metric]] = {
    'iou_score': smp.utils.metrics.IoU,
    'fscore': smp.utils.metrics.Fscore,
    'accuracy': smp.utils.metrics.Accuracy,
    'recall': smp.utils.metrics.Recall,
    'precision': smp.utils.metrics.Precision,
}
