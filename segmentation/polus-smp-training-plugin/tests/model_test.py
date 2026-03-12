import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

polus_dir = "/home/vihanimm/SegmentationModelToolkit/workdir/SMP_Pipeline/polus-plugins/segmentation/polus-smp-training-plugin/"
sys.path.append(polus_dir)



modelDir = "/home/vihanimm/SegmentationModelToolkit/workdir/output_SMP/Unet-MCCLoss-resnet18-random-Adam"
images_test_dir = (
    "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/image/"
)
labels_test_dir = "/home/vihanimm/SegmentationModelToolkit/Data/tif_data/nuclear/test/groundtruth_centerbinary_1pixelsmaller/"


def getLogs(logCSV: str, patience: int, log_dict: dict) -> dict:
    with open(logCSV) as csvfile:
        for row in list(csv.reader(csvfile))[-1 * patience :]:
            for parameter in row:
                parameter = parameter.split(":")
                key = parameter[0].strip(" ")
                value = float(parameter[1].strip(" "))
                log_dict[key]["all"].append(value)
                log_dict[key]["avg"] += value / patience
                log_dict[key]["mini"] = np.minimum(log_dict[key]["mini"], value)
                log_dict[key]["maxi"] = np.maximum(log_dict[key]["maxi"], value)

    return log_dict








# for metric in metrics:





# for test in test_loader:
#     for metric in metrics:
#         except:


class ModelTest(unittest.TestCase):
    def test_accuracy(self):
        print("acc")
        metric_name = "accuracy"
        metric_avg = metric_outputs[metric_name]["avg"]
        assert metric_avg > trainlog_dict[metric_name]["mini"]
        assert metric_avg > trainlog_dict[metric_name]["avg"] - trainlog_dict[metric_name]["avg"] * 0.1

    def test_iou_score(self):
        metric_name = "iou_score"
        metric_avg = metric_outputs[metric_name]["avg"]
        assert metric_avg > trainlog_dict[metric_name]["mini"]
        assert metric_avg > trainlog_dict[metric_name]["avg"] - trainlog_dict[metric_name]["avg"] * 0.1

    def test_fscore(self):
        metric_name = "fscore"
        metric_avg = metric_outputs[metric_name]["avg"]
        assert metric_avg > trainlog_dict[metric_name]["mini"]
        assert metric_avg > trainlog_dict[metric_name]["avg"] - trainlog_dict[metric_name]["avg"] * 0.1

    def test_recall(self):
        metric_name = "recall"
        metric_avg = metric_outputs[metric_name]["avg"]
        assert metric_avg > trainlog_dict[metric_name]["mini"]
        assert metric_avg > trainlog_dict[metric_name]["avg"] - trainlog_dict[metric_name]["avg"] * 0.1

    def test_precision(self):
        metric_name = "precision"
        metric_avg = metric_outputs[metric_name]["avg"]
        assert metric_avg > trainlog_dict[metric_name]["mini"]
        assert metric_avg > trainlog_dict[metric_name]["avg"] - trainlog_dict[metric_name]["avg"] * 0.1

    def test_loss(self):
        metric_name = loss
        metric_avg = metric_outputs[metric_name]["avg"]
        assert metric_avg > trainlog_dict[metric_name]["mini"]
        assert metric_avg > trainlog_dict[metric_name]["avg"] - trainlog_dict[metric_name]["avg"] * 0.1


if __name__ == "__main__":
    unittest.main()
