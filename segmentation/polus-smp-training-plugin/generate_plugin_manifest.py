import json

from src import utils

INPUTS = {
    'inferenceMode': {
        'description': '\'active\' or \'inactive\' for whether to run in inference mode.',
        'type': 'enum',
        'required': True,
        'options': {'values': ['active', 'inactive']},
    },
    'imagesInferenceDir': {
        'description': 'Collection containing images on which to run inference.',
        'type': 'collection',
    },
    'inferencePattern': {
        'description': 'Filename pattern for images on which to run inference.',
        'type': 'string',
    },

    'pretrainedModel': {
        'description': " ".join([
            'Path to a model that was previously trained with this plugin.',
            'If starting fresh, you must instead provide:',
            '\'modelName\',',
            '\'encoderBase\',',
            '\'encoderVariant\',',
            '\'encoderWeights\',',
            'and \'optimizerName\'.',
            'See the README for available options.'
        ]),
        'type': 'genericData',
    },
    'modelName': {
        'description': 'Model architecture to use. Required if starting fresh.',
        'type': 'enum',
        'options': {'values': utils.MODEL_NAMES},
    },
    'encoderBase': {
        'description': 'The name of the base encoder to use.',
        'type': 'enum',
        'options': {'values': utils.BASE_ENCODERS},
    },
    'encoderVariant': {
        'description': 'The name of the specific variant to use.',
        'type': 'enum',
        'options': {'values': utils.ENCODER_VARIANTS},
    },
    'encoderWeights': {
        'description': 'The name of the pretrained weights to use.',
        'type': 'enum',
        'options': {'values': list(sorted(utils.ENCODER_WEIGHTS))},
    },
    'optimizerName': {
        'description': (
            'Name of optimization algorithm to use for training the model. '
            'Required if starting fresh.'
        ),
        'type': 'enum',
        'options': {'values': utils.OPTIMIZER_NAMES},
    },

    'batchSize': {
        'description': (
            'Size of each batch for training. '
            'If left unspecified, we use the maximum possible based on memory constraints.'
        ),
        'type': 'number',
    },

    'imagesTrainDir': {
        'description': 'Collection containing images to use for training.',
        'type': 'collection',
    },
    'labelsTrainDir': {
        'description': 'Collection containing labels, i.e. the ground-truth, for the training images.',
        'type': 'collection',
    },
    'trainPattern': {
        'description': 'Filename pattern for training images and labels.',
        'type': 'string',
    },

    'imagesValidDir': {
        'description': 'Collection containing images to use for validation.',
        'type': 'collection',
    },
    'labelsValidDir': {
        'description': 'Collection containing labels, i.e. the ground-truth, for the validation images.',
        'type': 'collection',
    },
    'validPattern': {
        'description': 'Filename pattern for validation images and labels.',
        'type': 'string',
    },

    'device': {
        'description': 'Which device to use for the model',
        'type': 'string',
    },
    'checkpointFrequency': {
        'description': 'How often to save model checkpoints',
        'type': 'number',
    },

    'lossName': {
        'description': 'Name of loss function to use.',
        'type': 'enum',
        'options': {'values': utils.LOSS_NAMES},
    },
    'maxEpochs': {
        'description': 'Maximum number of epochs for which to continue training the model.',
        'type': 'number',
    },
    'patience': {
        'description': 'Maximum number of epochs to wait for model to improve.',
        'type': 'number',
    },
    'minDelta': {
        'description': 'Minimum improvement in loss to reset patience.',
        'type': 'number',
    },
}


OUTPUTS = [{
    'name': 'outputDir',
    'type': 'genericData',
    'description': 'In training mode, this contains the trained model and checkpoints. '
                   'In inference mode, this contains the output labels.'
}]

DEFAULTS = {
    'inferenceMode': 'inactive',
    'inferencePattern': '.*',
    'modelName': 'Unet',
    'encoderBase': 'ResNet',
    'encoderVariant': 'resnet34',
    'encoderWeights': 'imagenet',
    'optimizerName': 'Adam',
    'trainPattern': '.*',
    'validPattern': '.*',
    'device': 'gpu',
    'lossName': 'JaccardLoss',
    'maxEpochs': 100,
    'patience': 10,
    'minDelta': 1e-4,
}

INFERENCE_ARGS = {
    'imagesInferenceDir',
    'inferencePattern',
}

COMMON_ARGS = {
    'inferenceMode',
    'device',
    'pretrainedModel',
    'modelName',
    'encoderBase',
    'encoderVariant',
    'encoderWeights',
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
        minor = int(numbering[-1].split('debug')[0])
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

        if key not in COMMON_ARGS:
            if key != 'inferenceMode':
                if key in INFERENCE_ARGS:
                    field['condition'] = 'model.inputs.inferenceMode==active'
                else:
                    field['condition'] = 'model.inputs.inferenceMode==inactive'

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

    for key, value in INPUTS.items():
        if key != 'inferenceMode':
            value['required'] = False
        if 'options' not in value.keys():
            # noinspection PyTypeChecker
            value['options'] = None

    manifest = {
        'name': 'Demo SMP Training/Inference',
        'version': f'{version}',
        'title': 'Segmentation Models Training and Inference',
        'description': 'Segmentation models training and inference plugin.',
        'author': 'Gauhar Bains (gauhar.bains@labshare.org), Najib Ishaq (najib.ishaq@axleinfo.com), Madhuri Vihani (madhuri.vihani@nih.gov)',
        'institution': 'National Center for Advancing Translational Sciences, National Institutes of Health',
        'repository': 'https://github.com/PolusAI/polus-plugins/tree/dev/segmentation',
        'website': 'https://ncats.nih.gov/preclinical/core/informatics',
        'citation': '',
        'containerId': f'labshare/polus-smp-training-plugin::{version}',
        'inputs': [{'name': key, **value} for key, value in INPUTS.items()],
        'outputs': OUTPUTS,
        'ui': create_ui(),
        'validators': variants_conditionals() + weights_conditionals(),
    }

    with open('plugin.json', 'w') as outfile:
        json.dump(manifest, outfile, indent=4)

    return


if __name__ == '__main__':
    generate_manifest(debug=False)
