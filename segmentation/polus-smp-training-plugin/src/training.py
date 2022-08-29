import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from xmlrpc.client import Boolean

import albumentations as albu
import segmentation_models_pytorch as smp
import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
from filepattern import FilePattern
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.utils.base import Metric
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from segmentation_models_pytorch.utils.train import TrainEpoch
from segmentation_models_pytorch.utils.train import ValidEpoch

# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter

import utils

writer = SummaryWriter()
sys.path.append(os.path.dirname(__file__))

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("training")
logger.setLevel(utils.POLUS_LOG)


def initialize_model(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> Tuple[SegmentationModel, Optimizer]:
    """Initializes a model from a Checkpoint. A checkpoint knows the:

        * 'model_name': The architecture of the model in use.
            See utils.params.MODELS
        * 'encoder_variant': The name of the specific encoder architecture.
            See utils.params.ENCODERS
        * 'encoder_weights': The name of the dataset used to pre-train the
            encoder. See utils.params.ENCODERS
        * 'optimizer_name': The name of the optimization algorithm for training
            the model. See utils.params.OPTIMIZERS
        * 'final_epoch': The number of epochs for which the model has been
            trained by this plugin. 0 indicates a new model.
        * 'model_state_dict': Model state from a previous run of this plugin.
        * 'optimizer_state_dict': Optimizer state from a previous run of this
            plugin.

    Args:
        checkpoint: A Checkpoint dictionary.
        device: The device (gpu/cpu) on which to run the model.

    Returns:
        Instantiated SegmentationModel and the Optimizer

        If resuming training from a previous run of this plugin, the states of
            the model and optimizer are loaded in.
    """
    logger.info("Initializing model...")

    # noinspection PyArgumentList
    model = utils.MODELS[checkpoint["model_name"]](
        encoder_name=checkpoint["encoder_variant"],
        encoder_weights=checkpoint["encoder_weights"],
        in_channels=1,  # all images in WIPP are single-channel.
        activation="sigmoid",  # TODO: Change for Cellpose FlowFields
    )
    state_dict = checkpoint["model_state_dict"]
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.to(device)

    # noinspection PyArgumentList
    optimizer = utils.OPTIMIZERS[checkpoint["optimizer_name"]](
        params=model.parameters()
    )

    if checkpoint["final_epoch"] > 0:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer


def configure_augmentations():
    # TODO: These are the default augmentation we do for training.
    #  We will add WIPP arguments for configuring custom augmentations.
    transforms = [
        albu.RandomCrop(height=256, width=256),
        utils.PoissonTransform(peak=10, p=0.3),
        albu.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.4, p=0.2),
        albu.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.5, border_mode=0
        ),
        albu.PadIfNeeded(
            min_height=256, min_width=256, always_apply=True, border_mode=0
        ),
        albu.OneOf(
            [
                albu.MotionBlur(blur_limit=15, p=0.1),
                albu.Blur(blur_limit=15, p=0.1),
                albu.MedianBlur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
    ]

    return transforms


def initialize_dataloader(
    *,
    images_dir: Path,
    labels_dir: Optional[Path],
    pattern: str,
    batch_size: int,
    mode: str,
) -> TorchDataLoader:
    """Initializes a data-loaders for training or validation.

    TODO(Madhuri): Add docs

    Args:
        images_dir: Input Image collection.
        labels_dir: Labels collection for the input images.
        pattern: File-pattern for the images and labels.
        batch_size: Number of tiles per batch to use.
        mode: training, validation or inference

    Returns:
        A data-loader for training or validation.
    """
    assert mode in ("training", "validation", "inference")

    images_fp = FilePattern(images_dir, pattern)
    labels_fp = None if labels_dir is None else FilePattern(labels_dir, pattern)

    image_array, label_array = utils.get_labels_mapping(images_fp, labels_fp)
    dataset = utils.Dataset(
        images=image_array,
        labels=label_array,
        augs=configure_augmentations() if mode == "training" else None,
    )

    loader = utils.MultiEpochsDataLoader(
        dataset=dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    return loader


def initialize_epoch_iterators(
    *,
    model: SegmentationModel,
    loss: TorchLoss,
    metrics: List[Metric],
    device: torch.device,
    optimizer: Optimizer,
) -> Tuple[TrainEpoch, ValidEpoch]:
    """Initializes the training and validation iterators that train the model
        for each epoch.

    Args:
        model: The model being trained.
        loss: An instantiated Loss function for the model.
        metrics: A list of instantiated Metrics with which to track model performance.
        device: A torch device, wither a GPU or a CPU.
        optimizer: An instantiated optimizer with which to update the model.

    Returns:
        A 2-tuple of the epoch-iterators for training and validation.
    """
    logger.info("Initializing Epoch Iterators...")

    epoch_kwargs = dict(
        model=model, loss=loss, metrics=metrics, device=device, verbose=True
    )
    trainer = smp.utils.train.TrainEpoch(optimizer=optimizer, **epoch_kwargs)
    validator = smp.utils.train.ValidEpoch(**epoch_kwargs)

    return trainer, validator


def _log_epoch(
    logs: dict,
    file_path: Path,
    mode: str,
):
    logs: str = ", ".join(f"{k}: {v:.8f}" for k, v in logs.items())
    logger.info(f"{mode} logs: {logs}")
    with open(file_path, "a") as outfile:
        outfile.write(f"{str(logs)}\n")
    return


def batch_update_train(trainer, x, y):
    """
    TODO(Madhuri): Docs and type hints

    Args:
        trainer:
        x:
        y:

    Returns:

    """
    trainer.optimizer.zero_grad(set_to_none=True)
    prediction = trainer.model.forward(x)
    loss = trainer.loss(prediction, y)
    loss.backward()
    trainer.optimizer.step()
    return loss, prediction


def batch_update_valid(validator, x, y):
    """
    TODO(Madhuri): Docs and type hints

    Args:
        validator:
        x:
        y:

    Returns:

    """
    with torch.no_grad():
        prediction = validator.model.forward(x)
        loss = validator.loss(prediction, y)
    return loss, prediction


def start_training(
    epoch_iterators: Tuple[TrainEpoch, ValidEpoch],
    dataloaders: Tuple[TorchDataLoader, TorchDataLoader],
    early_stopping: Tuple[int, int, float],
    checkpoint: Dict[str, Any],
    checkpoint_frequency: int,
    output_dir: Path,
    prof=None,
) -> int:
    """
    TODO(Madhuri): Docs

    Args:
        epoch_iterators:
        dataloaders:
        early_stopping:
        checkpoint:
        checkpoint_frequency:
        output_dir:
        prof:

    Returns:

    """
    train_loader, valid_loader = dataloaders
    trainer, validator = epoch_iterators

    starting_epoch = checkpoint["final_epoch"]

    if checkpoint_frequency is not None:
        checkpoints_dir = output_dir.joinpath("checkpoints")
        checkpoints_dir.mkdir(parents=False, exist_ok=True)

    num_epochs, patience, min_delta = early_stopping
    best_loss = float("inf")
    epoch_index, epochs_without_improvement = starting_epoch + 1, 0

    for epoch_index, _ in enumerate(range(num_epochs), start=starting_epoch + 1):
        logger.info(
            "\t".join(
                (
                    5 * "-",
                    f"Epoch: {epoch_index}/{num_epochs + starting_epoch}",
                    5 * "-",
                )
            )
        )

        train_logs = {
            trainer.loss.__name__: AverageValueMeter(),
            **{metric.__name__: AverageValueMeter() for metric in trainer.metrics},
        }
        valid_logs = {
            validator.loss.__name__: AverageValueMeter(),
            **{metric.__name__: AverageValueMeter() for metric in validator.metrics},
        }

        for train_x, train_y in train_loader:  # iterating through the train batches
            train_x, train_y = train_x.to(trainer.device), train_y.to(trainer.device)
            train_loss, train_y_pred = batch_update_train(trainer, train_x, train_y)

            train_loss_value = train_loss.cpu().detach().numpy()
            train_logs[trainer.loss.__name__].add(train_loss_value)

            for train_metric_fn in trainer.metrics:
                train_metric_value = (
                    train_metric_fn(train_y_pred, train_y).cpu().detach().numpy()
                )
                train_logs[train_metric_fn.__name__].add(train_metric_value)

        train_logs = {k: v.mean for k, v in train_logs.items()}

        _log_epoch(
            logs=train_logs,
            file_path=output_dir.joinpath("trainlogs.csv"),
            mode="Train",
        )

        for valid_x, valid_y in valid_loader:  # iterating through the valid batches
            valid_x, valid_y = valid_x.to(validator.device), valid_y.to(
                validator.device
            )
            valid_loss, valid_y_pred = batch_update_valid(validator, valid_x, valid_y)

            valid_loss_value = valid_loss.cpu().detach().numpy()
            valid_logs[validator.loss.__name__].add(valid_loss_value)

            for valid_metric_fn in validator.metrics:
                valid_metric_value = (
                    valid_metric_fn(valid_y_pred, valid_y).cpu().detach().numpy()
                )
                valid_logs[valid_metric_fn.__name__].add(valid_metric_value)

        valid_logs = {k: v.mean for k, v in valid_logs.items()}

        _log_epoch(
            logs=valid_logs,
            file_path=output_dir.joinpath("validlogs.csv"),
            mode="Valid",
        )

        checkpoint.update(
            {
                "final_epoch": epoch_index,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
            }
        )

        current_loss = valid_logs[validator.loss.__name__]
        logger.info(f"CURRENT LOSS: {current_loss}")
        if (best_loss > current_loss) and (best_loss - current_loss >= min_delta):
            epochs_without_improvement = 0
            best_loss = current_loss
            torch.save(trainer.model, output_dir.joinpath("model.pth"))
            torch.save(checkpoint, output_dir.joinpath("checkpoint.pth"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    f"No improvement for {patience} epochs. Stopping training early..."
                )
                break

        logger.info(
            f"Epochs without Improvement: {epochs_without_improvement} of {patience}"
        )

        if checkpoint_frequency is not None:
            if (epoch_index % checkpoint_frequency) == 0:
                # noinspection PyUnboundLocalVariable
                torch.save(
                    trainer.model, checkpoints_dir.joinpath(f"model_{epoch_index}.pth")
                )
                torch.save(
                    checkpoint,
                    checkpoints_dir.joinpath(f"checkpoint_{epoch_index}.pth"),
                )

        if prof is not None:
            prof.step()

    else:
        logger.info(f"Finished training for user-specified {num_epochs} epochs...")

    if checkpoint_frequency is not None:
        torch.save(trainer.model, checkpoints_dir.joinpath("model_final.pth"))
        torch.save(checkpoint, checkpoints_dir.joinpath("checkpoint_final.pth"))

    return epoch_index


def train_model(
    *,
    dataloaders: Tuple[TorchDataLoader, TorchDataLoader],
    epoch_iterators: Tuple[TrainEpoch, ValidEpoch],
    early_stopping: Tuple[int, int, float],
    checkpoint: Dict[str, Any],
    checkpoint_frequency: int,
    output_dir: Path,
    tensorboard_profiler: Boolean,
) -> int:
    """Trains the model.

    Args:
        dataloaders: A 2-tuple of data-loaders for training and validation.
        epoch_iterators: A 2-tuple of iterators for training and validation.
        early_stopping: Criteria for cutting short model training. A 3-tuple of
            * the maximum number of epochs to train the model,
            * the maximum number of epochs to wait for the model to improve.
            * the minimum decrease in loss to consider an improvement.
        checkpoint: Dictionary containing information on the model checkpoints
        checkpoint_frequency: How often to save the model
        output_dir: The output directory to save model outputs
        tensorboard_profiler: TODO(Madhuri)

    Returns:
        The total number of epochs for which the model has been trained by this
            plugin.
    """

    # TODO(Najib): Figure out how this will work with WIPP outputs
    if tensorboard_profiler:
        tensorboard_dir = output_dir.joinpath("tensorboard")
        tensorboard_dir.mkdir(parents=False, exist_ok=True)
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(tensorboard_dir)
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:

            epoch = start_training(
                epoch_iterators,
                dataloaders,
                early_stopping,
                checkpoint,
                checkpoint_frequency,
                output_dir,
                prof=prof,
            )
    else:

        epoch = start_training(
            epoch_iterators,
            dataloaders,
            early_stopping,
            checkpoint,
            checkpoint_frequency,
            output_dir,
        )

    return epoch
