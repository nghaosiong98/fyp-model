import argparse
import os

import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T


def _get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',
                        help='Base model to be used for training',
                        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                        )
    parser.add_argument('--data', help='path to dataset')
    parser.add_argument('--output', help='path to keep log, figures, and checkpoints')
    return parser.parse_args()


def train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(short_edge_length=(512, 512), max_size=1333, sample_style='choice'),
        T.RandomFlip(),
        T.RandomBrightness(intensity_min=0.6, intensity_max=1.2),
    ]
    return augs


def setup():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.base_model))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MIN_SIZE_TRAIN = 512
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.base_model)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.OUTPUT_DIR = args.output
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"

    return cfg


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=train_aug(cfg))
        return build_detection_train_loader(cfg=cfg, mapper=mapper)


if __name__ == '__main__':
    args = _get_parsed_args()
    cfg = setup()
    for d in ["train", "val"]:
        register_coco_instances(
            d,
            {},
            os.path.join(args.data, d, "annotations.json"),
            os.path.join(args.data, d),
        )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("val", cfg, False, output_dir=args.output)
    val_loader = build_detection_test_loader(cfg, "val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
