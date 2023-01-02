import os
import json
import random
import shutil
from copy import deepcopy
from typing import Any, Dict, Tuple

import cv2
from tqdm import tqdm

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json
from detectron2.utils.visualizer import Visualizer


TRAIN_DATA_NAME = 'acod_train'
TRAIN_JSON_FILE = '/home/summerlee/Documents/data/acod/train_coco/new_annotations_2.json'
TRAIN_IMAGE_ROOT = '/home/summerlee/Documents/data/acod/train_coco'
VAL_DATA_NAME = 'acod_val'
VAL_JSON_FILE = '/home/summerlee/Documents/data/acod/valid_coco/new_annotations_2.json'
VAL_IMAGE_ROOT = '/home/summerlee/Documents/data/acod/valid_coco'


def register_acod(
    train_data_name: str = TRAIN_DATA_NAME,
    train_json_file: os.PathLike = TRAIN_JSON_FILE,
    train_image_root: os.PathLike = TRAIN_IMAGE_ROOT,
    val_data_name: str = VAL_DATA_NAME,
    val_json_file: os.PathLike = VAL_JSON_FILE,
    val_image_root: os.PathLike = VAL_IMAGE_ROOT
    
) -> None:
    register_coco_instances(train_data_name, {}, train_json_file, train_image_root)
    register_coco_instances(val_data_name, {}, val_json_file, val_image_root)
    
    
def filter_acod_json(save_path: os.PathLike, json_file: os.PathLike, image_root: os.PathLike) -> None:
    with open(json_file, 'r') as f:
        pre_json = json.load(f)
        
    print(f"len(pre_json['images']): {len(pre_json['images'])}")
    print(f"len(pre_json['annotations']): {len(pre_json['annotations'])}")
    print(f"len(os.listdir(os.path.join(image_root, 'JPEGImages'))): {len(os.listdir(os.path.join(image_root, 'JPEGImages')))}")
        
    new_json = deepcopy(pre_json)

    new_images = []
    image_ids = []
    for json_image in tqdm(pre_json['images']):
        image_path = os.path.join(image_root, json_image['file_name'])
        image_id = json_image['id']
        
        if os.path.exists(image_path):
            new_images.append(json_image)
            image_ids.append(image_id)
        else:
            print(image_path)
            
    new_annotations = []
    for json_annot in tqdm(pre_json['annotations']):
        if json_annot['image_id'] in image_ids:
            new_annotations.append(json_annot)
            
    new_json['images'] = new_images
    new_json['annotations'] = new_annotations
    new_json['categories'] = pre_json['categories'][1:]
    print(new_json['categories'], '\n')
    
    print(f"len(new_json['images']): {len(new_json['images'])}")
    print(f"len(new_json['annotations']): {len(new_json['annotations'])}")
    
    with open(save_path, 'w') as f:
        json.dump(new_json, f)
        
        
def filter_train_acod_json(
    save_path: os.PathLike = TRAIN_JSON_FILE,
    json_file: os.PathLike = '/workspace/summer-lee/data/acod/train_coco/new_annotations.json',
    image_root: os.PathLike = TRAIN_IMAGE_ROOT,
) -> None:
    filter_acod_json(save_path=save_path, json_file=json_file, image_root=image_root)


def filter_val_acod_json(
    save_path: os.PathLike = VAL_JSON_FILE,
    json_file: os.PathLike = '/workspace/summer-lee/data/acod/valid_coco/new_annotations.json',
    image_root: os.PathLike = VAL_IMAGE_ROOT,
) -> None:
    filter_acod_json(save_path=save_path, json_file=json_file, image_root=image_root)


def verify_acod(
    save_dir: os.PathLike,
    json_file: os.PathLike,
    image_root: os.PathLike,
    dataset_name: str,
    seed: int = 36
) -> Tuple[Dict[str, Any]]:
    # print(os.getcwd(), save_dir)
    # check_dir(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    
    dataset_dicts = load_coco_json(json_file, image_root, dataset_name)
    acod_metadata = MetadataCatalog.get(dataset_name)
    for d in random.Random(seed).sample(dataset_dicts, 3):
        # print(d['file_name'])
        img = cv2.imread(d['file_name'])
        visualizer = Visualizer(img[:, :, ::-1], metadata=acod_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite(os.path.join(save_dir, os.path.split(d['file_name'])[-1]), out.get_image()[:, :, ::-1])
        
    return dataset_dicts, acod_metadata


def verify_train_acod(
    save_dir: os.PathLike,
    json_file: os.PathLike = TRAIN_JSON_FILE,
    image_root: os.PathLike = TRAIN_IMAGE_ROOT,
    dataset_name: str = TRAIN_DATA_NAME,
) -> Tuple[Dict[str, Any]]:
    return verify_acod(save_dir=save_dir, json_file=json_file, image_root=image_root, dataset_name=dataset_name)


def verify_val_acod(
    save_dir: os.PathLike,
    json_file: os.PathLike = VAL_JSON_FILE,
    image_root: os.PathLike = VAL_IMAGE_ROOT,
    dataset_name: str = VAL_DATA_NAME,
) -> Tuple[Dict[str, Any]]:
    return verify_acod(save_dir=save_dir, json_file=json_file, image_root=image_root, dataset_name=dataset_name)
