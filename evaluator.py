# filepath: /root/TFVSN/evaluator.py
"""
评估器模块: 包含用于视频摘要评估的函数和类
"""

import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple


def calculate_f1_score(predicted_summary: np.ndarray, 
                     user_summary: np.ndarray, 
                     eval_method: str = "avg") -> float:
    """
    计算F1分数
    
    Args:
        predicted_summary: 预测的摘要
        user_summary: 用户标注的摘要
        eval_method: 评估方法，"avg"使用平均值，"max"使用最大值
        
    Returns:
        F1分数
    """
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    
    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        # 计算precision, recall, f-score
        precision = sum(overlapped) / (sum(S) + 1e-8)
        recall = sum(overlapped) / (sum(G) + 1e-8)
        
        if precision + recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))
            
    if eval_method == "max":
        return max(f_scores)
    else:  # "avg"
        return sum(f_scores) / len(f_scores)


def calculate_precision_recall(predicted_summary: np.ndarray, 
                              ground_truth: np.ndarray) -> Dict[str, float]:
    """
    计算精确率和召回率
    
    Args:
        predicted_summary: 预测的摘要
        ground_truth: 真实标注的摘要
        
    Returns:
        包含精确率和召回率的字典
    """
    max_len = max(len(predicted_summary), ground_truth.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    
    precision_values = []
    recall_values = []
    
    for user in range(ground_truth.shape[0]):
        G[:ground_truth.shape[1]] = ground_truth[user]
        overlapped = S & G
        
        # 计算precision, recall
        precision = sum(overlapped) / (sum(S) + 1e-8)
        recall = sum(overlapped) / (sum(G) + 1e-8)
        
        precision_values.append(precision * 100)
        recall_values.append(recall * 100)
            
    return {
        "precision": sum(precision_values) / len(precision_values),
        "recall": sum(recall_values) / len(recall_values)
    }


class VideoSummarizationEvaluator:
    """
    视频摘要评估器，用于评估视频摘要质量
    """
    
    def __init__(self, ground_truth: np.ndarray, predicted_summary: np.ndarray, eval_method: str = "avg"):
        """
        初始化视频摘要评估器
        
        Args:
            ground_truth: 真实标注的摘要
            predicted_summary: 预测的摘要
            eval_method: 评估方法，"avg"使用平均值，"max"使用最大值
        """
        self.ground_truth = ground_truth
        self.predicted_summary = predicted_summary
        self.eval_method = eval_method

    def evaluate_f1(self) -> float:
        """
        评估F1分数
        
        Returns:
            F1分数
        """
        return calculate_f1_score(self.predicted_summary, self.ground_truth, self.eval_method)
        
    def evaluate_precision_recall(self) -> Dict[str, float]:
        """
        评估精确率和召回率
        
        Returns:
            包含精确率和召回率的字典
        """
        return calculate_precision_recall(self.predicted_summary, self.ground_truth)
    
    def evaluate_comprehensive(self) -> Dict[str, float]:
        """
        综合评估，包含F1分数、精确率和召回率
        
        Returns:
            包含多种评估指标的字典
        """
        f1_score = self.evaluate_f1()
        pr_metrics = self.evaluate_precision_recall()
        
        return {
            "f1_score": f1_score,
            **pr_metrics
        }
    
    def evaluate(self, metric_type: str = "f1") -> Union[float, Dict[str, float]]:
        """
        根据指定的度量类型进行评估
        
        Args:
            metric_type: 评估指标类型，可以是 "f1"、"precision_recall" 或 "comprehensive"
            
        Returns:
            根据指定类型返回相应的评估结果
        """
        if metric_type == "f1":
            return self.evaluate_f1()
        elif metric_type == "precision_recall":
            return self.evaluate_precision_recall()
        elif metric_type == "comprehensive":
            return self.evaluate_comprehensive()
        else:
            raise ValueError(f"不支持的评估指标类型: {metric_type}")
