"""Mesmer Training."""
import enum
import errno
import logging
import os
import pathlib

import filepattern
import numpy as np
from bfio import BioReader
from deepcell import image_generators, losses
from deepcell.model_zoo.panopticnet import PanopticNet
from deepcell.utils.data_utils import reshape_matrix
from deepcell.utils.train_utils import count_gpus, get_callbacks, rate_scheduler
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

seed = 0  # seed for random train-test split
logger = logging.getLogger("training")
logger.setLevel(logging.INFO)


class BACKBONES(str, enum.Enum):
    """Keras models."""

    FEATURENET = "featurenet"
    FEATURENET3D = "featurenet3d"
    FEATURENET_3D = "featurenet_3d"
    DENSENET121 = "densenet121"
    DENSENET169 = "densenet169"
    DENSENET201 = "densenet201"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"
    RESNET50V2 = "resnet50v2"
    RESNET101V2 = "resnet101v2"
    RESNET152V2 = "resnet152v2"
    VGG16 = "vgg16"
    VGG19 = "vgg19"
    NASNET_LARGE = "nasnet_large"
    NASNET_MOBILE = "nasnet_mobile"
    MOBILENET = "mobilenet"
    MOBILENETV2 = "mobilenetv2"
    MOBILENET_V2 = "mobilenet_v2"
    EFFICIENTNETB0 = "efficientnetb0"
    EFFICIENTNETB1 = "efficientnetb1"
    EFFICIENTNETB2 = "efficientnetb2"
    EFFICIENTNETB3 = "efficientnetb3"
    EFFICIENTNETB4 = "efficientnetb4"
    EFFICIENTNETB5 = "efficientnetb5"
    EFFICIENTNETB6 = "efficientnetb6"
    EFFICIENTNETB7 = "efficientnetb7"
    EFFICIENTNETV2B0 = "efficientnetv2b0"
    EFFICIENTNETV2B1 = "efficientnetv2b1"
    EFFICIENTNETV2B2 = "efficientnetv2b2"
    EFFICIENTNETV2B3 = "efficientnetv2b3"
    EFFICIENTNETV2BL = "efficientnetv2bl"
    EFFICIENTNETV2BM = "efficientnetv2bm"
    EFFICIENTNETV2BS = "efficientnetv2bs"
    DEFAULT = "resnet50"


class MesmerTrain:
    """Training a Mesmer segmentation model.

    Args:
        xtrain_path: Training images.
        ytrain_path : Training label images.
        xtest_path: Test images.
        ytest_path: Test label images.
        file_pattern: Pattern to parse file names.
        tile_size: Input image tile size.
        model_backbone: Use Keras models as DeepCell backbones which can be instantiated with weights pretrained on ImageNet.
        iterations: Number of training iterations.
        batch_size: Number of images processed to update the model.
        out_dir: Path to output directory.
    """

    def __init__(
        self,
        xtrain_path: pathlib.Path,
        ytrain_path: pathlib.Path,
        xtest_path: pathlib.Path,
        ytest_path: pathlib.Path,
        model_backbone: str,
        file_pattern: str,
        tile_size: int,
        iterations: int,
        batch_size: int,
        out_dir: pathlib.Path,
    ):
        """Define Instance attributes."""
        self.xtrain_path = xtrain_path
        self.ytrain_path = ytrain_path
        self.xtest_path = xtest_path
        self.ytest_path = ytest_path
        self.file_pattern = file_pattern
        self.tile_size = tile_size
        self.model_backbone = model_backbone
        self.iterations = iterations
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.ROOT_DIR = pathlib.Path("/data")
        self.MODEL_DIR = pathlib.Path(self.ROOT_DIR, "models", self.out_dir)
        self.LOG_DIR = pathlib.Path(self.ROOT_DIR, "logs", self.out_dir)
        self.create_directories()
        model_name = "watershed_centroid_nuclear_general_std"
        self.model_path = os.path.join(self.MODEL_DIR, f"{model_name}.h5")
        self.loss_path = os.path.join(self.MODEL_DIR, f"{model_name}.npz")

    def create_directories(self) -> None:
        """Create directories."""
        for d in (self.MODEL_DIR, self.LOG_DIR):
            try:
                os.makedirs(d)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def get_data(self, x):
        """Prepare a list of image data."""
        data = []
        fp = filepattern.FilePattern(x, self.file_pattern)
        for file in fp:
            tile_grid_size = 1
            tile_size = tile_grid_size * 2048
            with BioReader(file[1][0]) as br:
                for z in range(br.Z):
                    for y in range(0, br.Y, tile_size):
                        y_max = min([br.Y, y + tile_size])
                        for x in range(0, br.X, tile_size):
                            x_max = min([br.X, x + tile_size])
                            im = np.squeeze(
                                br[y:y_max, x:x_max, z : z + 1, 0, 0]  # noqa
                            )
                            im = np.expand_dims(im, 2)
                            data.append(im)
        return data

    def semantic_loss(self, n_classes: int):
        """Create a loss function for each semantic head."""

        def _semantic_loss(y_true, y_pred):
            if n_classes > 1:
                return 0.01 * losses.weighted_categorical_crossentropy(
                    y_true, y_pred, n_classes=n_classes
                )
            return MSE(y_true, y_pred)

        return _semantic_loss

    def run(self) -> None:
        """Run the training segmentation model."""
        X_train = np.array(self.get_data(self.xtrain_path))
        y_train = np.array(self.get_data(self.ytrain_path))
        X_test = np.array(self.get_data(self.xtest_path))
        y_test = np.array(self.get_data(self.ytest_path))
        logger.info("input loaded...")

        X_train, y_train = reshape_matrix(X_train, y_train, reshape_size=self.tile_size)
        X_test, y_test = reshape_matrix(X_test, y_test, reshape_size=self.tile_size)
        print(f"X.shape: {X_train.shape}\ny.shape: {y_train.shape}")

        classes = {
            "inner_distance": 1,  # inner distance
            "outer_distance": 1,  # outer distance
            "fgbg": 2,  # foreground/background separation
        }

        model = PanopticNet(
            backbone="efficientnetb0",
            input_shape=X_train.shape[1:],
            norm_method="std",
            num_semantic_classes=classes,
            location=True,  # should always be true
            include_top=True,
        )

        # norm_method = 'whole_image'  # data normalization
        lr = 1e-5
        optimizer = Adam(lr=lr, clipnorm=0.001)
        lr_sched = rate_scheduler(lr=lr, decay=0.99)
        batch_size = int(self.batch_size)
        min_objects = 1  # throw out images with fewer than this many objects
        transforms = list(classes.keys())
        transforms_kwargs = {"outer-distance": {"erosion_width": 0}}
        # use augmentation for training but not validation
        datagen = image_generators.SemanticDataGenerator(
            rotation_range=180,
            shear_range=0,
            zoom_range=(0.75, 1.25),
            horizontal_flip=True,
            vertical_flip=True,
        )

        datagen_val = image_generators.SemanticDataGenerator(
            rotation_range=0,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=0,
            vertical_flip=0,
        )

        train_data = datagen.flow(
            {"X": X_train, "y": y_train},
            seed=seed,
            transforms=transforms,
            transforms_kwargs=transforms_kwargs,
            min_objects=min_objects,
            batch_size=batch_size,
        )

        val_data = datagen_val.flow(
            {"X": X_test, "y": y_test},
            seed=seed,
            transforms=transforms,
            transforms_kwargs=transforms_kwargs,
            min_objects=min_objects,
            batch_size=batch_size,
        )

        loss = {}
        # Give losses for all of the semantic heads
        for layer in model.layers:
            if layer.name.startswith("semantic_"):
                n_classes = layer.output_shape[-1]
                loss[layer.name] = self.semantic_loss(n_classes)
        model.compile(loss=loss, optimizer=optimizer)
        num_gpus = count_gpus()
        print("Training on", num_gpus, "GPUs.")
        train_callbacks = get_callbacks(
            self.model_path,
            lr_sched=lr_sched,
            tensorboard_log_dir=self.LOG_DIR,
            save_weights_only=num_gpus >= 0,
            monitor="val_loss",
            verbose=1,
        )
        model.fit(
            train_data,
            steps_per_epoch=train_data.y.shape[0] // self.batch_size,
            epochs=self.iterations,
            validation_data=val_data,
            validation_steps=val_data.y.shape[0] // self.batch_size,
            callbacks=train_callbacks,
        )
