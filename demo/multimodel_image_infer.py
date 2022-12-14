import os
import os.path as osp
from pathlib import Path
import glob
import re
import numpy as np
import cv2 as cv
import json
import pickle
from copy import deepcopy
from argparse import ArgumentParser
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def make_save_path(savedir, name="exp"):
    save_dir = increment_path(Path(savedir) / name, exist_ok=False)
    print(f"The resault save to {save_dir}\n")

    return save_dir


def save_new(savedir, imgname, shapes, imgshape):
    save_dir = osp.join(savedir, "vote_new")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    
    savepath = osp.join(save_dir, imgname.replace("jpg", "json"))
    new_set = {
        "version": "",
        "flags": {},
        "shapes": shapes,
        "imagePath": imgname,
        "imageData": None,
        "imageWidth": imgshape[1],
        "imageHeight": imgshape[0]
    }
    # print(f"save new: {savepath}")
    with open(savepath, "w", encoding="utf8") as fp:
        json.dump(new_set, fp, indent=2)


def load_pth_file(filepath):
    with open(filepath, "rb") as fp:
        pth_data = pickle.load(fp)

    return pth_data


def write_pth_file(data, filepath):
    with open(filepath, "wb") as fp:
        pickle.dump(data, fp)


def make_shapes(all_cotegory_bboxes):
    shapes = []
    for cotegory_id in range(len(all_cotegory_bboxes)):
        labels = all_cotegory_bboxes[cotegory_id]
        for label in labels:
            for cotegory_name, bbox in label.items():
                bbox = [float(i) for i in bbox]
                shape = {
                    "label": cotegory_name,
                    "is_verify": None,
                    "points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                shapes.append(shape)
                # print(shape)
    
    return shapes


def nms_bboxes(results, classes, score_thr):
    all_cotegory_bboxes = []
    for cotegory_name, cotegory_id in classes.items():
        for ids, bbox in enumerate(results):
            bboxes = []
            if ids == int(cotegory_id):
                for point in bbox:
                    if point[-1] > score_thr:
                        bboxes.append({cotegory_name: point})
        
            all_cotegory_bboxes.append(bboxes)

    return all_cotegory_bboxes


def areaIoU(fpoint, spoint, vote_score):
    score_1 = fpoint[-1]*vote_score[0]
    score_2 = spoint[-1]*vote_score[1]
    if score_1 > score_2:
        bbox = fpoint
    else:
        bbox = spoint

    x11, y11, x12, y12 = fpoint[0:4]
    x21, y21, x22, y22 = spoint[0:4]
    
    x_min = np.max([x11, x21])
    x_max = np.min([x12, x22])
    y_min = np.max([y11, y21])
    y_max = np.min([y12, y22])

    inter_area = (x_max - x_min) * (y_max - y_min)
    if x_max-x_min < 0 or y_max-y_min < 0:
        inter_area = 0
    
    box1_area = (x12-x11)*(y12-y11)
    box2_area = (x22-x21)*(y22-y21)

    union_area = box1_area + box2_area - inter_area
    area_IoU = inter_area / union_area

    return bbox, area_IoU


def weight_vote(multi_results, vote_scores, th=0.8):
    # print(f"multi_results: {len(multi_results[0])}")
    new_result = []
    for one_results in zip(*multi_results):
        cotegorys_list = []
        # print(f"one_results: {len(one_results)}")
        results_nums = len(one_results)

        for i in range(results_nums-1):
            for flabels in one_results[i]:
                for slabels in one_results[i+1]:
                    for cotegory_name, fbbox in flabels.items():
                        for cotegory_name, sbbox in slabels.items():
                            bbox, comIoU = areaIoU(fbbox, sbbox, vote_scores[i:i+2])
                            # print(f"comIoU: {comIoU}")
                            if comIoU > th:
                                cotegorys_list.append({cotegory_name: bbox})


        new_result.append(cotegorys_list)
        # print(f"new_result: {new_result}")
        
    return new_result


def multi_infer(configs):
    img_dir = configs["image_dir"]
    config_list = configs["config_infos"]["configs"]
    checkpoints_list = configs["config_infos"]["checkpoints"]
    out_dir = configs["out_dir"]
    pre_result = configs["pre_result_file"]
    device = configs["device"]
    score_thr_list = configs["score_thr"]
    vote_score_list = configs["vote_score"]
    classes = configs["classes"]
    print(configs)

    savedir = make_save_path(out_dir)
    os.makedirs(savedir)

    img_list = sorted(os.listdir(img_dir))
    # img_nums = len(img_list)
    
    multi_result_list = []
    multi_result_file = osp.join(out_dir, pre_result)
    if osp.exists(multi_result_file):
        print(f"from {multi_result_file} load data")
        # with open(multi_result_file, "rb") as fp:
        #    multi_result_list = pickle.load(fp)
        process_results = load_pth_file(multi_result_file)
    else:
        count = 0
        for config, checkpoint in zip(config_list, checkpoints_list):
            model = init_detector(config, checkpoint, device=device)
            one_result_list = []
            for  index, filename in enumerate(img_list):
                img_path = osp.join(img_dir, filename)
                print(f'{checkpoint} model ({count+1}/{len(config_list)}) is processing the {index+1}/{len(img_list)} images.')
                result = inference_detector(model, img_path)
                filter_bboxes = nms_bboxes(result,  classes, score_thr_list[count])
                
                one_result_list.append(filter_bboxes)
            
            multi_result_list.append(one_result_list)
            count += 1
        process_results = {
            "configs": configs,
            "data": multi_result_list
            }
        write_pth_file(process_results, multi_result_file)
        # with open(multi_result_file, "wb") as fp:
        #     pickle.dump(multi_result_list, fp)
        print(f"seccessful save the data to {multi_result_file}")

    multi_result_list = process_results["data"]
    print("multi len: ", len(multi_result_list))
    count = 0
    for multi_result in zip(*multi_result_list):
        # print(f"multi_result: {len(multi_result)}")
        # print(f"multi_result: {multi_result}")

        voted_bboxes = weight_vote(multi_result, vote_score_list)
        shapes = make_shapes(voted_bboxes)
        
        filename = img_list[count]
        img_path = osp.join(img_dir, filename)
        img = cv.imread(img_path)
        imgshape = img.shape
        save_new(savedir, filename, shapes, imgshape)
        print(f"save result in {savedir} {count+1}/{len(img_list)}")
        count += 1


def get_args(args):
    filepath = args.configs_path
    with open(filepath, "r", encoding="utf8") as fp:
        configs = json.load(fp)

    return configs


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--configs_path', default="./demo/configs.json", help='')
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    configs = get_args(args)
    
    multi_infer(configs)
