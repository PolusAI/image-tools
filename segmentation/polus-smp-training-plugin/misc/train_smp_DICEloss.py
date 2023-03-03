# rudimentary libraries for basic commands
import json, os, sys
import copy

# most important library
import segmentation_models_pytorch as smp


# pytorch functions
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

# update to this for multiprocessing
from torch.utils.data import IterableDataset
import albumentations as albu

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import fbeta_score, jaccard_score

from tqdm import tqdm
from tqdm import trange

CUDA_LAUNCH_BLOCKING = 1

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# for pytorch, you need to create an abstract Dataset Class
class DatasetforPytorch(Dataset):

    def __init__(self, 
                images_dir,
                masks_dir,
                preprocessing=None,
                augmentations=None):

        self.images_fps = [os.path.join(images_dir, image) for image in os.listdir(images_dir)]
        self.masks_fps =  [os.path.join(masks_dir, mask)   for mask in os.listdir(masks_dir)]
        self.preprocessing = preprocessing # this is a function that is getting intialized
        self.augmentations = augmentations # this is a function that is getting initialized

    def __getitem__(self, i):

        image          = np.array(Image.open(self.images_fps[i]))
        mask           = np.array(Image.open(self.masks_fps[i]))
        # mask[mask > 0] = 1 

        if self.augmentations:
            sample = self.augmentations(image=image, mask=mask)
            image, mask = sample['image'], sample['mask'] 

        image          = np.reshape(image, (1, image.shape[0], image.shape[1])).astype("float32")
        assert np.isnan(image).any() == False
        assert np.isinf(image).any() == False

        
        mask           = np.reshape(mask, (1, mask.shape[0], mask.shape[1])).astype("float32")
        assert np.isnan(mask).any() == False
        assert np.isinf(mask).any() == False



        return image, mask

    def __len__(self):
        return(len(self.images_fps))

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Make sure inputs are probits

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        albu.RandomCrop(height=256, width=256, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(256, 256)
    ]
    return albu.Compose(test_transform)

def plot_fromnohup(file_name):

    train_log_losses = []
    train_log_iou    = []
    train_log_f      = []
    valid_log_losses = []
    valid_log_iou    = []
    valid_log_f      = []
    nohupfile = "/home/vihanimm/SegmentationModelToolkit/workdir/pytorch_binary/nohup.out"
    with open(nohupfile, 'r') as nohup:
        for line in nohup:
            line = line.rstrip()
            if line.startswith("TRAIN EPOCH"):
                line = line.split(":")
                line = line[1]
                loss, f, iou = line.split(",")
                loss = float((loss.lstrip().split(" "))[1])
                f    = float((f.lstrip().split(" "))[2])
                iou  = float((iou.lstrip().split(" "))[2])
                train_log_losses.append(loss)
                train_log_iou.append(iou)
                train_log_f.append(f)
            elif line.startswith("VALID EPOCH"):
                line = line.split(":")
                line = line[1]
                loss, f, iou = line.split(",")
                loss = float((loss.lstrip().split(" "))[1])
                f    = float((f.lstrip().split(" "))[2])
                iou  = float((iou.lstrip().split(" "))[2])
                valid_log_losses.append(loss)
                valid_log_iou.append(iou)
                valid_log_f.append(f)
            else:
                continue

    fig, axs = plt.subplots(2, 3, figsize=(24, 16), tight_layout=True)
    
    axs[0,0].plot(train_log_losses)
    axs[0,0].set_title("Training Data - Dice Loss")
    axs[0,0].set_ylabel("BCE Loss")
    axs[0,1].plot(train_log_iou)
    axs[0,1].set_title("Training Data - IOU Score")
    axs[0,1].set_ylabel("IOU Score")
    axs[0,2].plot(train_log_f)
    axs[0,2].set_title("Training Data - F Score")
    axs[0,2].set_ylabel("F Score")

    axs[1,0].plot(valid_log_losses)
    axs[1,0].set_title("Validation Data - Dice Loss")
    axs[1,0].set_ylabel("BCE Loss")
    axs[1,1].plot(valid_log_iou)
    axs[1,1].set_title("Validation Data - IOU Score")
    axs[1,1].set_ylabel("IOU Score")
    axs[1,2].plot(valid_log_f)
    axs[1,2].set_title("Validation Data - F Score")
    axs[1,2].set_ylabel("F Score")

    fig.suptitle("Loss, Fscore, IOU Scores")
    for x in axs.flat:
        x.set(xlabel='EPOCHS')
    

    plt.savefig(file_name)

def plot_histories(train_logs, valid_logs, file_name):
    
    fig, axs = plt.subplots(2, 3, figsize=(24, 16), tight_layout=True)
    
    axs[0,0].plot(train_logs["losses"])
    axs[0,0].set_title("Training Data - BCE Loss")
    axs[0,0].set_ylabel("BCE Loss")
    axs[0,1].plot(train_logs["iou_scores"])
    axs[0,1].set_title("Training Data - IOU Score")
    axs[0,1].set_ylabel("IOU Score")
    axs[0,2].plot(train_logs["f_scores"])
    axs[0,2].set_title("Training Data - F Score")
    axs[0,2].set_ylabel("F Score")

    axs[1,0].plot(valid_logs["losses"])
    axs[1,0].set_title("Validation Data - BCE Loss")
    axs[1,0].set_ylabel("BCE Loss")
    axs[1,1].plot(valid_logs["iou_scores"])
    axs[1,1].set_title("Validation Data - IOU Score")
    axs[1,1].set_ylabel("IOU Score")
    axs[1,2].plot(valid_logs["f_scores"])
    axs[1,2].set_title("Validation Data - F Score")
    axs[1,2].set_ylabel("F Score")

    fig.suptitle("Loss, Fscore, IOU Scores")

    for x in axs.flat:
        x.set(xlabel='EPOCHS')
    

    plt.savefig(file_name)


def train(data_directory          : str,
          groundtruth_basedirname : str,
          device                  : str,
          model_file              : str,
          patience                : int):


    # get the training images and masks
    train_directory = os.path.join(data_directory, "train")
    x_train_dir     = os.path.join(train_directory, "image")
    y_train_dir     = os.path.join(train_directory, groundtruth_basedirname)

    validation_directory = os.path.join(data_directory, "validation")
    x_valid_dir          = os.path.join(validation_directory, "image")
    y_valid_dir          = os.path.join(validation_directory, groundtruth_basedirname)

    # model building prerequistes
    UNET_basicmodel = smp.Unet(in_channels=1,
                               encoder_weights = None)

    model = UNET_basicmodel.to(device)
    model.train()
    print(summary(model, input_size=(1,512,512)))

    train_dataset = DatasetforPytorch(images_dir=x_train_dir, masks_dir=y_train_dir, 
                        augmentations=get_training_augmentation())
    valid_dataset = DatasetforPytorch(images_dir=x_valid_dir, masks_dir=y_valid_dir,
                        augmentations=get_validation_augmentation())

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    loss       = DiceLoss()
    fscore_fxn = smp.utils.metrics.Fscore(threshold=0.5)
    iou_fxn    = smp.utils.metrics.IoU(threshold=0.5)
    sig        = nn.Sigmoid()

    optimizer = torch.optim.Adam(UNET_basicmodel.parameters(), lr=0.0001) # only calculate for the parameters specified
    # tqdm_train_loader = tqdm(train_loader)
    # tqdm_valid_loader = tqdm(valid_loader)

    train_logs_list = {"losses": [], "f_scores": [], "iou_scores": []}
    valid_logs_list = {"losses": [], "f_scores": [], "iou_scores": []}

    # relevant for the while loop
    max_score = 0
    epoch     = 0
    early_stopping_counter = 0

    stop_the_training = False
    while stop_the_training == False:

        epoch_loss   = 0
        epoch_iou    = 0
        epoch_fscore = 0
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # clear all data from optimizer.step()
            output = model(data)
            probability = sig(output)
            assert torch.isnan(output).any() == False
            assert torch.isinf(output).any() == False
            losses = loss.forward(probability, target) # take inputs, and pass thru till we get to the 
                # numbers we want to optimize, which is the loss function (losses.item())
            fscore = fscore_fxn.forward(probability, target)
            iou    = iou_fxn.forward(probability, target)
            # tqdm_train_loader.set_description(f"LOSS: {losses.item()}, F SCORE {fscore.item()}, IOU SCORE {iou.item()}")
            epoch_loss   += losses.item()/len(train_loader)
            epoch_fscore += fscore.item()/len(train_loader)
            epoch_iou    += iou.item()/len(train_loader)
            losses.backward() # applying back propagation, cacluating the gradients/derivatives. 
            optimizer.step() # this updates weights. 
            

        train_logs_list["losses"].append(epoch_loss)
        train_logs_list["f_scores"].append(epoch_fscore)
        train_logs_list["iou_scores"].append(epoch_iou)
        print(f"TRAIN EPOCH {epoch}: Loss {epoch_loss}, F Score {epoch_fscore}, Iou Score {epoch_iou}") 

        epoch_loss   = 0
        epoch_iou    = 0
        epoch_fscore = 0
        for (data, target) in valid_loader:
            data, target = data.to(device), target.to(device)     
            optimizer.zero_grad()
            output = model(data)
            probability = sig(output)
            assert torch.isnan(output).any() == False
            assert torch.isinf(output).any() == False
            losses = loss(probability, target)
            fscore = fscore_fxn.forward(probability, target)
            iou    = iou_fxn.forward(probability, target)
            # tqdm_valid_loader.set_description(f"LOSS: {losses.item()}, F SCORE {fscore.item()}, IOU SCORE {iou.item()}")
            epoch_loss   += losses.item()/len(valid_loader)
            epoch_fscore += fscore.item()/len(valid_loader)
            epoch_iou    += iou.item()/len(valid_loader)
            losses.backward()
            optimizer.step()

        valid_logs_list["losses"].append(epoch_loss)
        valid_logs_list["f_scores"].append(epoch_fscore)
        valid_logs_list["iou_scores"].append(epoch_iou)
        print(f"VALID EPOCH {epoch}: Loss {epoch_loss}, F Score {epoch_fscore}, Iou Score {epoch_iou}") 

        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

        if epoch_fscore > max_score:
            max_score = epoch_fscore
            torch.save(UNET_basicmodel, model_file)
            print("MODEL SAVED with F Score of {}".format(epoch_fscore))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1 # then add one to the counter
            print(f"EARLY STOPPING COUNTER PLUS ONE: {early_stopping_counter}")
            if early_stopping_counter >= patience:
                stop_the_training = True
            
        if (epoch%100) == 0:
            torch.save(UNET_basicmodel, model_file[:-4] + f"_{epoch}.pth")

        epoch += 1
        print(" ")

    return train_logs_list, valid_logs_list


def visualize_output(model_pathway,
                     data_directory,
                     device,
                     output_dir,
                     groundtruth_basedir):
    
    def add_to_axis(image, groundtruth, threshold, axis=None):

        
        new_img = copy.deepcopy(image)
        new_img[new_img < threshold]  = 0
        new_img[new_img >= threshold] = 1

        f1_score = fbeta_score(y_true=groundtruth, 
                               y_pred=new_img,
                               average=None,
                               beta=1, zero_division='warn')
        f1_score = np.around(np.average(f1_score), 4)

        j_score  = jaccard_score(y_true=groundtruth,
                                 y_pred=new_img,
                                 average=None, zero_division='warn')
        j_score  = np.around(np.average(j_score), 4)
        # print(f1_score, j_score)

        if axis != None:
            axis.imshow(new_img)
            axis.set_title(f"Threshold: {threshold} - F1: {f1_score}, JACCARD: {j_score}")

        return f1_score, j_score
    
    sig = nn.Sigmoid()
    
    best_model = torch.load(model_pathway)
    model_name = os.path.basename(model_pathway)
    model_info = model_name.split("_")
    encoder    = "resnet34"
    encoder_weights = "imagenet"

    test_directory = os.path.join(data_directory, "test")
    x_test_dir     = os.path.join(test_directory, "image")
    y_test_dir     = os.path.join(test_directory, groundtruth_basedir)
    num_images     = len(os.listdir(x_test_dir))

    # preprocess_input = smp.encoders.get_preprocessing_fn(encoder, pretrained='imagenet')
    
    # Dataset for visualizing
    test_dataset_vis = DatasetforPytorch(images_dir=x_test_dir, masks_dir=y_test_dir)

    # create test dataset
    test_dataset = preprocessing=DatasetforPytorch(images_dir=x_test_dir, masks_dir=y_test_dir)

    nums = [958, 1148, 574, 698, 811, 925]
    max_f1 = {}
    max_j  = {}
    for i in tqdm(range(len(nums))):
        # n = np.random.choice(len(test_dataset_vis))
        n = nums[i]
        # n = i

        image_vis = test_dataset_vis[n][0]
        image, gt_mask = test_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        # print(np.unique(gt_mask))
        
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = sig(pr_mask) # need to make predicitions range from 0 to 1
        pr_mask = pr_mask.squeeze().cpu().numpy()
        pr_mask_shape = pr_mask.shape

        # print("IMAGE, GROUNDTRUTH, and PREDICTED unique values, respectively")
        # print(np.unique(image))
        # print(np.unique(gt_mask))
        # print(np.unique(pr_mask))

        fig, ((ax_img, ax_groundtruth, ax_prediction), 
            (ax_pred1, ax_pred2, ax_pred3), 
            (ax_pred4, ax_pred5, ax_pred6),
            (ax_pred7, ax_pred8, ax_pred9))= plt.subplots(4, 3, figsize = (24, 24))

        ax_img.imshow(image_vis.squeeze())
        ax_img.set_title("Image")
        ax_groundtruth.imshow(gt_mask.squeeze())
        ax_groundtruth.set_title("Groundtruth")
        ax_prediction.imshow(pr_mask)
        ax_prediction.set_title("Prediction Channel 0")

        f1_score_1, j_score_1 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.1, axis=ax_pred1)
        f1_score_2, j_score_2 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.2, axis=ax_pred2)
        f1_score_3, j_score_3 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.3, axis=ax_pred3)
        f1_score_4, j_score_4 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.4, axis=ax_pred4)
        f1_score_5, j_score_5 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.5, axis=ax_pred5)
        f1_score_6, j_score_6 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.6, axis=ax_pred6)
        f1_score_7, j_score_7 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.7, axis=ax_pred7)
        f1_score_8, j_score_8 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.8, axis=ax_pred8)
        f1_score_9, j_score_9 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.9, axis=ax_pred9)

        # f1_score_1, j_score_1 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.1) # axis=ax_pred1)
        # f1_score_2, j_score_2 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.2) # axis=ax_pred1)
        # f1_score_3, j_score_3 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.3) # axis=ax_pred1)
        # f1_score_4, j_score_4 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.4) # axis=ax_pred1)
        # f1_score_5, j_score_5 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.5) # axis=ax_pred1)
        # f1_score_6, j_score_6 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.6) # axis=ax_pred1)
        # f1_score_7, j_score_7 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.7) # axis=ax_pred1)
        # f1_score_8, j_score_8 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.8) # axis=ax_pred1)
        # f1_score_9, j_score_9 = add_to_axis(image=pr_mask, groundtruth=gt_mask, threshold=0.9) # axis=ax_pred1)

        list_threshold_f1 = [f1_score_1, f1_score_2, f1_score_3, f1_score_4, \
                             f1_score_5, f1_score_6, f1_score_7, f1_score_8, f1_score_9]
        list_threshold_j = [j_score_1, j_score_2, j_score_3, j_score_4, \
                             j_score_5, j_score_6, j_score_7, j_score_8, j_score_9]
        threshold_max_f1 = max(list_threshold_f1)
        threshold_max_j  = max(list_threshold_j)
        max_f1[i] = threshold_max_f1
        max_j[i]  = threshold_max_j

        fig.suptitle(f"Testing Image {n}")
        plot_name = os.path.join(output_dir, f"testingimage_{n}")
        plt.savefig(plot_name)

    average_f1 = np.average(list(max_f1.values()))
    average_j  = np.average(list(max_j.values()))

    max_f1 = sorted(max_f1.items(), key=lambda kv: kv[1])
    max_j  = sorted(max_j.items(), key=lambda kv: kv[1])

    print(average_f1, average_j)
    print("MAX F1 Sorted")
    print(max_f1)
    print("MAX J Sorted")
    print(max_j)


def main():
        
    # parameters
    object_identified = "cell"
    device            = "cuda:0"

    data_basedirectory  = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/"
    groundtruth_basedir = "groundtruth_centerbinary"
    file_directory      = os.path.dirname(os.path.abspath(__file__))
    output_model_directory     = os.path.join(file_directory, "models")
    if not os.path.isdir(output_model_directory):
        os.mkdir(output_model_directory)

    assert os.path.isdir(output_model_directory), f"{output_model_directory} is not an existing directory for models"
    assert os.path.isdir(data_basedirectory), f"{data_basedirectory} is not an existing directory for data"
    assert os.path.isdir(file_directory), f"{file_directory} is not existing directory for {__file__}"

    output_model_file  = os.path.join(output_model_directory, "UNET_BinaryDiceLoss.pth")
    output_train_json  = os.path.join(output_model_directory, "{}_train.json".format(output_model_file[:-4]))
    output_valid_json  = os.path.join(output_model_directory, "{}_valid.json".format(output_model_file[:-4]))
    output_score_graph = os.path.join(output_model_directory, "{}.jpg".format(output_model_file[:-4]))

    data_directory = os.path.join(data_basedirectory, object_identified)
    assert os.path.isdir(data_directory), "f{data_directory} does not exist"

    train_history, valid_history = train(data_directory          = data_directory,
                                            groundtruth_basedirname = groundtruth_basedir,
                                            device                  = device,
                                            model_file              = output_model_file,
                                            patience                = 10)

    json.dump(train_history, open(output_train_json, 'w'))
    json.dump(valid_history, open(output_valid_json, 'w'))

    # plot_fromnohup(output_score_graph)

    plot_histories(train_logs = train_history, 
                    valid_logs = valid_history,
                    file_name = output_score_graph)

    # output_model_file = "/home/vihanimm/SegmentationModelToolkit/workdir/pytorch_binary/models/UNET_BinaryDiceLoss_1800.pth"
    visualize_output(model_pathway       = output_model_file, 
                     data_directory      = data_directory,
                     device              = device,
                     output_dir          = file_directory,
                     groundtruth_basedir = groundtruth_basedir)

main()


