import re
import json
import math
import pdb

def seg_thinking_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>\s*<d_think>.*?</d_think>" # WISE
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0

def seg_segmentation_format_reward(predict_str: str) -> float:
    def is_valid_format(predict_str: str) -> bool:
        try:
            json_match = re.search(r'{[^}]+}', predict_str)
            if not json_match:
                return False
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # check the required keys
            required_keys = ['bbox', 'points_1', 'points_2']
            for key in required_keys:
                if key not in data:
                    return False
            
            # check the format of the value
            bbox = data['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                return False
                
            points_1 = data['points_1']
            points_2 = data['points_2']
            if not isinstance(points_1, list) or len(points_1) != 2:
                return False
            if not isinstance(points_2, list) or len(points_2) != 2:
                return False

            return True
        except Exception:
            return False
    return 1.0 if is_valid_format(predict_str) else 0.0

def seg_iou_reward(predict_str: str, ground_truth: str) -> float:
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        area1 = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
        area2 = (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
        union = area1 + area2 - inter
        return float(inter)/union
    
    try:
        ground_truth = ground_truth.strip()
        gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        gt_match = re.search(gt_box_pattern, ground_truth)
        if gt_match:
            gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), int(gt_match.group(3)), int(gt_match.group(4))]
            
        json_pattern = r'{[^}]+}'  
        json_match = re.search(json_pattern, predict_str)
        # pdb.set_trace()
        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                # if iou(content_bbox, gt_bbox) > 0.5:
                #     return 1.0
                return iou(content_bbox, gt_bbox)  # TODO
    except Exception:
        pass
    return 0.0


def seg_box_l1_reward(predict_str: str, ground_truth: str) -> float:
    def l1_distance(box1, box2):
        return (abs(box1[0]-box2[0]) + abs(box1[1]-box2[1]) + abs(box1[2]-box2[2]) + abs(box1[3]-box2[3])) / 4
    
    try:
        ground_truth = ground_truth.strip()
        gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        gt_match = re.search(gt_box_pattern, ground_truth)
        if gt_match:
            gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), int(gt_match.group(3)), int(gt_match.group(4))]
            
        json_pattern = r'{[^}]+}'  
        json_match = re.search(json_pattern, predict_str)
        if json_match:
            data = json.loads(json_match.group(0))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
                if l1_distance(content_bbox, gt_bbox) < 10:
                    return 1.0
    except Exception:
        pass
    return 0.0

def seg_point_l1_reward(predict_str: str, ground_truth: str) -> float:
    def points_in_box(point, bbox):
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
    
    def points_distance(points1, points2):
        dist1 = math.sqrt((points1[0][0]-points2[0][0])**2 + (points1[0][1]-points2[0][1])**2) + \
                math.sqrt((points1[1][0]-points2[1][0])**2 + (points1[1][1]-points2[1][1])**2)
        
        dist2 = math.sqrt((points1[0][0]-points2[1][0])**2 + (points1[0][1]-points2[1][1])**2) + \
                math.sqrt((points1[1][0]-points2[0][0])**2 + (points1[1][1]-points2[0][1])**2)
        return min(dist1, dist2) / 2
        
    try: 
        gt_points_pattern = r'<points>\((\d+),(\d+)\),\((\d+),(\d+)\)</points>'
        gt_match = re.search(gt_points_pattern, ground_truth)
        if gt_match:
            gt_points = [[int(gt_match.group(1)), int(gt_match.group(2))], [int(gt_match.group(3)), int(gt_match.group(4))]]
            
        json_pattern = r'{[^}]+}' 
        json_match = re.search(json_pattern, predict_str)

        if json_match:
            data = json.loads(json_match.group(0))
            # find bbox key
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]
            # find points key
            points_keys = ['points_1', 'points_2']  # get the first two points keys
            if len(points_keys) == 2:
                point1 = data[points_keys[0]]
                point2 = data[points_keys[1]]
                point1 = [int(point1[0]), int(point1[1])]
                point2 = [int(point2[0]), int(point2[1])]
                if points_in_box(point1, content_bbox) and points_in_box(point2, content_bbox):
                    if points_distance([point1, point2], gt_points) < 100:
                        return 1.0
    except Exception:
        pass  # Continue to next verification method if this fails
    return 0.0

def extract_pattern(pattern: str, predict_str: str):
    match = re.search(pattern, predict_str, re.DOTALL)
    if match:
        return match.group(0)
    return ""


from sentence_transformers import SentenceTransformer, util

# 2. 加载一个预训练好的、强大的句子编码模型
# 'all-mpnet-base-v2' 是一个性能非常高的通用模型
# 'all-MiniLM-L6-v2' 是一个更小、更快的模型，性能也很不错
name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(name)
print(f'[INFO] {name} have loaded...')

def sim_score(d_think, think):
    embed1 = model.encode(d_think, convert_to_tensor=True)
    embed2 = model.encode(think, convert_to_tensor=True)
    score = util.cos_sim(embed1, embed2).item()
    return score

def seg_d_think_reward(correct, predict_str):
    if not correct:
        return {"dist_reward": 0}

    def calculate_distillation_reward(d_think: str, think: str) -> float:
        """
        计算思考蒸馏奖励，同时考虑语义相似度和精简度。
        R = Sim_Score * Concise_Score
        """
        if not d_think or not think:
            return 0.0

        # 1. 计算语义相似分
        sim_score_value = sim_score(d_think, think)
        
        # for after have L1
        d_think_len = token_count(d_think)
        think_len = token_count(think)
        
        # 防止 d_think_len 为0导致除零错误
        if d_think_len == 0:
            return 0.0
            
        conciseness_score = max(0.0, 1.0 - (think_len / d_think_len))
        
        # 3. 最终奖励是两者的乘积
        final_reward = sim_score_value * conciseness_score
        
        return final_reward
        
    score = 0.0
    try:
        d_think = extract_pattern("<d_think>.*?</d_think>", predict_str)
        think = extract_pattern("<think>.*?</think>", predict_str)
        score = calculate_distillation_reward(d_think, think)
    except Exception as e:
        print(f"[Error] in seg_d_think_reward: {e}")
    return {"dist_reward": score}
    




# >>> for L1

from transformers import AutoTokenizer
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
def token_count(text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)

def l1_exact_reward(correct: bool,
                    n_target: int,
                    n_gen: int,
                    alpha: float = 0.0003) -> float:
    if not correct:
        return 0.0

    length_error = abs(n_target - n_gen)
    reward = 1.0 - alpha * length_error
    return max(0.0, reward)      # 防止出现负数
import math

def l1_max_reward(correct: bool,
                  n_target: int,
                  n_gen: int,
                  alpha: float = 0.0003,
                  delta: float = 0.5) -> float:
    if not correct:
        return 0.0

    # 计算长度差值
    length_diff = n_target - n_gen
    raw_score = alpha * length_diff + delta

    # 将分数裁剪到 [0, 1]
    reward = max(0.0, min(1.0, raw_score))
    return reward


def seg_len_L1_reward(correct, predict_str):
    score = 0.0
    try:
        think = extract_pattern("<think>.*?</think>", predict_str)
        think_len = token_count(think)
        # score = l1_exact_reward(correct, 57, think_len)
        score = l1_max_reward(correct, 20, think_len)
    except Exception as e:
        print(f"[Error] in seg_len_L1_reward: {e}")
    return {"l1_max_20_reward": score}

# <<<

def seg_strict_compute_score(predict_str: str, ground_truth: str) -> float:
    
    if "<think> streamlined thinking process here </think>" in predict_str \
        or "<d_think> detailed thinking process here </d_think>" in predict_str:
        return {"reward": 0}
    thinking_format_reward = seg_thinking_format_reward(predict_str)
    
    segmentation_format_reward = seg_segmentation_format_reward(predict_str)
    iou_reward = seg_iou_reward(predict_str, ground_truth)
    point_l1_reward = seg_point_l1_reward(predict_str, ground_truth)
    box_l1_reward = seg_box_l1_reward(predict_str, ground_truth)
    
    reward = {
        "thinking_format_reward": thinking_format_reward,
        "segmentation_format_reward": segmentation_format_reward,
        "iou_reward": iou_reward,
        "point_l1_reward": point_l1_reward,
        "box_l1_reward": box_l1_reward,
    }
    
    correct = iou_reward > 0.5
    d_think_reward = seg_d_think_reward(correct, predict_str)
    # len_l1_reward = seg_len_L1_reward(correct, predict_str)
    
    reward.update(d_think_reward)
    return reward