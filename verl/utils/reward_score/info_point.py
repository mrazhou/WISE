import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple

import numpy as np
import math
from itertools import combinations
from typing import List, Tuple

import json
import re

def get_bbox_diagonal(info_map: np.ndarray) -> float:
    """
    计算二值掩码的边界框的对角线长度。
    这是应该在【数据预处理】阶段调用的函数。
    """
    y_indices, x_indices = np.where(info_map == 1)
    if len(x_indices) == 0:
        return 0.0
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    
    return math.sqrt(bbox_w**2 + bbox_h**2)

def calculate_point_diversity_reward(
    predicted_points: List[Tuple[int, int]],
    normalization_diagonal: float
) -> float:
    """
    计算点集的多样性（分散程度）奖励。
    使用一个给定的对角线长度进行归一化。

    Args:
        predicted_points: 模型预测的点坐标列表。
        normalization_diagonal: 用于归一化的对角线长度（推荐使用物体边界框的对角线）。

    Returns:
        float: 归一化后的多样性奖励分数。
    """
    num_points = len(predicted_points)
    if num_points < 2:
        return 0.0

    total_distance = sum(
        math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        for p1, p2 in combinations(predicted_points, 2)
    )
    
    average_distance = total_distance / (num_points * (num_points - 1) / 2)
    
    if normalization_diagonal == 0:
        return 0.0

    diversity_reward = average_distance / normalization_diagonal
    # 确保奖励不会因为预测点超出边界框而大于1
    return min(diversity_reward, 1.0)

def points_in_box(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

def calculate_info_reward(
    informativeness_map: np.ndarray,
    predicted_points: List[Tuple[int, int]]
) -> float:
    """
    根据给定的信息量地图和预测点，计算奖励分数。
    这是应该在【强化学习在线训练】阶段调用的函数。

    Args:
        informativeness_map (np.ndarray): 由 create_map_from_mask 生成的地图。
        predicted_points (List[Tuple[int, int]]): 模型预测的点坐标列表，
                                                    格式为 [(x1, y1), (x2, y2), ...]。

    Returns:
        float: 最终计算出的平均奖励分数。
    """
        
    if not predicted_points:
        return 0.0
        
    score_sum = 0.0
    map_h, map_w = informativeness_map.shape

    for p in predicted_points:
        x, y = p
        # 检查坐标是否越界，并使用整数索引
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < map_h and 0 <= ix < map_w:
            score_sum += informativeness_map[iy, ix]
    
    return score_sum / len(predicted_points)



def calculate_reward(info_map, predict_str):
    try:
        # answer = r'<answer>(.*?)</answer>'  
        # answer_match = re.search(answer, predict_str)
        # if answer_match:
        #     data = json.loads(answer_match.group(1))
        json_pattern = r'{[^}]+}'
        json_match = re.search(json_pattern, predict_str)
        # pdb.set_trace()
        if json_match:
            data = json.loads(json_match.group(0))
            points_keys = ['points_1', 'points_2']
            point1 = data[points_keys[0]]
            point2 = data[points_keys[1]]
            point1 = [int(point1[0]), int(point1[1])]
            point2 = [int(point2[0]), int(point2[1])]
            
            bbox_key = 'bbox'
            content_bbox = data[bbox_key]
            if points_in_box(point1, content_bbox) and points_in_box(point2, content_bbox):
                predicted_points = [point1, point2]
            else:
                return {
                    "info_reward": 0.0,
                    "diversity_reward": 0
                }
        else:
            return {
                "info_reward": 0.0,
                "diversity_reward": 0
            }
    except Exception as e:
        print(f"An error occurred during calculate reward: {e}")
        return {
                "info_reward": 0,
                "diversity_reward": 0
            }

    info_map = np.array(info_map)

    normalization_diagonal = get_bbox_diagonal(info_map)
    diversity_reward = calculate_point_diversity_reward(predicted_points, normalization_diagonal)
    info_reward = calculate_info_reward(info_map, predicted_points)
    
    return {
        "info_reward": info_reward,
        "diversity_reward": diversity_reward
    }





# def calculate_reward(info_map, predicted_points):

#     normalization_diagonal = get_bbox_diagonal(info_map)
#     diversity_reward = calculate_point_diversity_reward(predicted_points, normalization_diagonal)
#     info_reward = calculate_info_reward(info_map, predicted_points)
    
#     return {
#         "info_reward": info_reward,
#         "diversity_reward": diversity_reward
#     }