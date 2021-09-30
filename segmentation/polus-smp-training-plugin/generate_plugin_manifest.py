import json

from src import utils

INPUTS = {
    'pretrainedModel': {
        'description': (
            'Path to a model that was previously trained with this plugin. '
            'If starting fresh, you must instead provide: '
            '\'modelName\', '
            '\'encoderBaseVariantWeights\', and '
            '\'optimizerName\'. '
            'See the README for available options.'
        ),
        'type': 'genericData',
        'required': False,
    },
    'modelName': {
        'description': 'Model architecture to use. Required if starting fresh.',
        'type': 'enum',
        'required': False,
        'options': {'values': utils.MODEL_NAMES},
    },
    'encoderBase': {
        'description': 'The name of the base encoder to use.',
        'type': 'enum',
        'required': False,
        'options': {'values': utils.BASE_ENCODERS},
    },
    'encoderVariant': {
        'description': 'The name of the specific variant to use.',
        'type': 'enum',
        'required': False,
        'options': {'values': utils.ENCODER_VARIANTS},
    },
    'encoderWeights': {
        'description': 'The name of the pretrained weights to use.',
        'type': 'enum',
        'required': False,
        'options': {'values': list(sorted(utils.ENCODER_WEIGHTS))},
    },
    'optimizerName': {
        'description': (
            'Name of optimization algorithm to use for training the model. '
            'Required if starting fresh.'
        ),
        'type': 'enum',
        'required': False,
        'options': {'values': utils.OPTIMIZER_NAMES},
    },

    'batchSize': {
        'description': (
            'Size of each batch for training. '
            'If left unspecified, we use the maximum possible based on memory constraints.'
        ),
        'type': 'number',
        'required': False,
    },

    'imagesDir': {
        'description': 'Collection containing images.',
        'type': 'collection',
        'required': True,
    },
    'imagesPattern': {
        'description': 'Filename pattern for images.',
        'type': 'string',
        'required': True,
    },
    'labelsDir': {
        'description': 'Collection containing labels, i.e. the ground-truth, for the images.',
        'type': 'collection',
        'required': True,
    },
    'labelsPattern': {
        'description': 'Filename pattern for labels.',
        'type': 'string',
        'required': True,
    },
    'trainFraction': {
        'description': 'Fraction of dataset to use for training.',
        'type': 'number',
        'required': False,
    },
    'segmentationMode': {
        'description': 'The kind of segmentation to perform.',
        'type': 'enum',
        'required': False,
        'options': {
            'values': ['binary', 'multilabel', 'multiclass'],
        },
    },

    'lossName': {
        'description': 'Name of loss function to use.',
        'type': 'enum',
        'required': False,
        'options': {'values': utils.LOSS_NAMES},
    },
    'metricName': {
        'description': 'Name of performance metric to track.',
        'type': 'enum',
        'required': False,
        'options': {'values': utils.METRIC_NAMES},
    },
    'maxEpochs': {
        'description': 'Maximum number of epochs for which to continue training the model.',
        'type': 'number',
        'required': False,
    },
    'patience': {
        'description': 'Maximum number of epochs to wait for model to improve.',
        'type': 'number',
        'required': False,
    },
    'minDelta': {
        'description': 'Minimum improvement in loss to reset patience.',
        'type': 'number',
        'required': False,
    },
}

OUTPUTS = [{
    'name': 'outputDir',
    'type': 'genericData',
    'description': 'Output model and checkpoint.'
}]

DEFAULTS = {
    'modelName': 'Unet',
    'encoderBase': 'ResNet',
    'encoderVariant': 'resnet34',
    'encoderWeights': 'imagenet',
    'optimizerName': 'Adam',
    'trainFraction': 0.7,
    'lossName': 'JaccardLoss',
    'metricName': 'IoU',
    'maxEpochs': 100,
    'patience': 10,
    'minDelta': 1e-4,
}


def bump_version(debug: bool) -> str:
    with open('VERSION', 'r') as infile:
        version = infile.read()

    if debug:
        if 'debug' in version:
            [version, debug] = version.split('debug')
            version = f'{version}debug{str(1 + int(debug))}'
        else:
            version = f'{version}debug1'
    else:
        numbering = version.split('.')
        minor = int(numbering[-1])
        minor += 1
        numbering[-1] = str(minor)
        version = '.'.join(numbering)

    with open('VERSION', 'w') as outfile:
        outfile.write(version)

    return version


def create_ui():
    ui = list()

    for key, values in INPUTS.items():
        field = {
            'key': f'inputs.{key}',
            'title': key,
            'description': values['description'],
        }

        if key in DEFAULTS:
            field['default'] = DEFAULTS[key]

        ui.append(field)

    return ui


def variants_conditionals():
    validator = list()
    for base, variant in utils.ENCODERS.items():
        validator.append({
            'condition': [{
                'input': 'encoderBase',
                'value': base,
                'eval': '==',
            }],
            'then': [{
                'action': 'show',
                'input': 'encoderVariant',
                'values': list(variant.keys()),
            }]
        })
    return validator


def weights_conditionals():
    validator = list()

    for base, variants in utils.ENCODERS.items():
        for variant, weights in variants.items():
            validator.append({
                'condition': [{
                    'input': 'encoderVariant',
                    'value': variant,
                    'eval': '==',
                }],
                'then': [{
                    'action': 'show',
                    'input': 'encoderWeights',
                    'values': [*weights, 'random'],
                }],
            })

    return validator


def generate_manifest(debug: bool):
    version = bump_version(debug)
    # noinspection PyTypeChecker
    manifest = {
        'name': 'SegmentationModelsTraining',
        'version': f'{version}',
        'title': 'SegmentationModelsTraining',
        'description': 'Segmentation models training plugin',
        'author': 'Gauhar Bains (gauhar.bains@labshare.org), Najib Ishaq (najib.ishaq@axleinfo.com)',
        'institution': 'National Center for Advancing Translational Sciences, National Institutes of Health',
        'repository': 'https://github.com/polus-au/polus-plugins-dl',
        'website': 'https://ncats.nih.gov/preclinical/core/informatics',
        'citation': '',
        'containerId': f'labshare/polus-smp-training-plugin::{version}',
        'inputs': [{'name': key, **value} for key, value in INPUTS.items()],
        'outputs': OUTPUTS,
        'ui': create_ui(),
        'validators': variants_conditionals() + weights_conditionals()
    }

    with open('plugin.json', 'w') as outfile:
        json.dump(manifest, outfile, indent=4)

    return


if __name__ == '__main__':
    generate_manifest(debug=True)
