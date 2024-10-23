from ultralytics.utils.instance import Instances
from typing import Dict, List

import os
import json
import random

import numpy as np
import cv2


def get_xyxy_bbox(instance):

    return [
        instance["box"]["xmin"],
        instance["box"]["ymin"],
        instance["box"]["xmax"],
        instance["box"]["ymax"]
    ]


class SyntheticDataMixture:

    def __init__(self, synthetic_images_root_dir: str,
                 synthetic_labels_root_dir: str,
                 synthetic_labelmap: Dict[str, int] = {},
                 synthetic_prob: float = 0.5):

        self.synthetic_images_root_dir = synthetic_images_root_dir
        self.synthetic_labels_root_dir = synthetic_labels_root_dir
        self.synthetic_labelmap = synthetic_labelmap
        self.synthetic_prob = synthetic_prob

        self.synthetic_images = os.listdir(
            synthetic_images_root_dir
        )
        self.synthetic_labels = os.listdir(
            synthetic_labels_root_dir
        )

    def __call__(self, labels: dict) -> dict:

        if random.random() < self.synthetic_prob:

            synthetic_idx = random.randint(0, len(self.synthetic_images) - 1)

            synthetic_image = self.synthetic_images[synthetic_idx]
            synthetic_label = self.synthetic_labels[synthetic_idx]

            synthetic_image_path = os.path.join(
                self.synthetic_images_root_dir,
                synthetic_image
            )

            synthetic_label_path = os.path.join(
                self.synthetic_labels_root_dir,
                synthetic_label
            )

            synthetic_image = cv2.imread(
                synthetic_image_path
            )

            with open(synthetic_label_path, "r") as synthetic_labels_file:

                synthetic_labels = json.load(
                    synthetic_labels_file
                )

            synthetic_classes = [
                self.synthetic_labelmap.get(instance["label"], 0)
                for instance in synthetic_labels
            ]

            synthetic_bboxes = [
                get_xyxy_bbox(instance)
                for instance in synthetic_labels
            ]

            synthetic_instances = Instances(
                bboxes = np.array(synthetic_bboxes),
                bbox_format = "xyxy"
            )

            labels["img"] = synthetic_image
            labels["cls"] = synthetic_classes
            labels["instances"] = synthetic_instances

        return labels