import time
import os

import cv2
import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from register_acod import TRAIN_DATA_NAME, register_acod


SAMPLE_VIDEO_PATH = '/home/summerlee/Documents/backup-213/github-repositories/summer_yolact/data/construction-videos/2007.mp4'


def eval_video(video_path: os.PathLike = SAMPLE_VIDEO_PATH) -> None:
    print('start setup...')
    
    cap = cv2.VideoCapture(video_path)

    target_fps   = round(cap.get(cv2.CAP_PROP_FPS))
    frame_width  = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter('Output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (frame_width, frame_height))
    
    register_acod()
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))  # r50
    # cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))  # r101
    
    cfg.DATASETS.TRAIN = (TRAIN_DATA_NAME,)
    
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = 'mask_rcnn_R_50_FPN_3x/model_0005999.pth'  # r50
    # cfg.MODEL.WEIGHTS = 'model_0003999.pth'  # r101
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    
    predictor = DefaultPredictor(cfg)
    
    print('setup finished.')
    
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
    
        if ret:
            start_time = time.time()
            
            # main process
            outputs = predictor(frame)
            
            end_time = time.time()

            fps = 1 / (end_time - start_time)
            print(f'FPS: {fps:.1f}, frame: {i+1}/{num_frames}', end='\r')
            
            v = Visualizer(frame[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)
            v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            v = v.get_image()[:, :, ::-1]
            out.write(v)
            
            i += 1

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break
    
    print('\nfinish')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    exit()


def main():
    eval_video()


if __name__ == '__main__':
    main()