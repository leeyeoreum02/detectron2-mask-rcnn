import argparse
from contextlib import redirect_stdout
import os
import random

import cv2

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from register_acod import VAL_DATA_NAME, register_acod, verify_val_acod
from train_acod_mask_rcnn import NUM_CLASSES, seed_everything


def eval(args) -> None:
    register_acod()
    dataset_dicts, acod_metadata = verify_val_acod('examples/acod_val')
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config_path))
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = args.weights  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.roi_score_thresh_test  # set a custom testing threshold
    
    predictor = DefaultPredictor(cfg)
    
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d['file_name'])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=acod_metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite(os.path.join(args.examples_dir, os.path.split(d['file_name'])[-1]), out.get_image()[:, :, ::-1])
    
    get_coco_metrics(cfg, args, predictor)


def get_coco_metrics(cfg, args, predictor) -> None:
    evaluator = COCOEvaluator(VAL_DATA_NAME, output_dir=args.output_dir)
    val_loader = build_detection_test_loader(cfg, VAL_DATA_NAME)
    
    with open(os.path.join(args.output_dir, 'coco_metrics.txt'), 'w') as f:
        with redirect_stdout(f):
            print(inference_on_dataset(predictor.model, val_loader, evaluator))


def main() -> None:
    seed_everything()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--weights', default='model_0009999.pth')
    parser.add_argument('--roi_score_thresh_test', type=float, default=0.3)    
    parser.add_argument('--examples_dir', default='examples/outputs')
    parser.add_argument('--output_dir', required=True)

    args = parser.parse_args()
    
    eval(args)


if __name__ == '__main__':
    main()
