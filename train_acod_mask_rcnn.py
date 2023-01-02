import argparse
from contextlib import redirect_stdout
import os
import random

import numpy as np
import wandb

import torch
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator

from register_acod import TRAIN_DATA_NAME, VAL_DATA_NAME, register_acod, verify_train_acod, verify_val_acod


NUM_CLASSES = 10


torch.multiprocessing.set_sharing_strategy('file_system')


def seed_everything(seed: int = 36):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name=VAL_DATA_NAME):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)


def train(args) -> None:
    register_acod()
    verify_train_acod('examples/acod_train')
    verify_val_acod('examples/acod_val')
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config_path))
    cfg.DATASETS.TRAIN = (TRAIN_DATA_NAME,)
    cfg.DATASETS.TEST = (VAL_DATA_NAME,)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config_path)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size  # This is the real "batch size" commonly known to deep learning people
    # cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = 58000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = [38667, 51556]  # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_head_batch_size  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SOLVER.CHECKPOINT_PERIOD = args.eval_period

    cfg.OUTPUT_DIR = args.output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(cfg.OUTPUT_DIR, 'config.txt'), 'w') as f:
        with redirect_stdout(f):
            print(cfg)
    
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    
    
def get_args() -> argparse.Namespace:
    parser = default_argument_parser()
    
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--roi_head_batch_size', type=int, default=128)
    parser.add_argument('--eval_period', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu_ids', type=str, default='0', help='delimeter is `,`.')
    
    return parser


def main(args) -> None:
    seed_everything()
    
    wandb.init(project='JCDE-2022-444', sync_tensorboard=True, entity='summerlee', name=args.output_dir)
    
    train(args)
    
    wandb.finish()


if __name__ == '__main__':
    args = get_args().parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    args.num_gpus = len(args.gpu_ids.split(','))
    # args.num_gpus = 10
    # args.dist_url = 'auto'
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
