"""
模型评估模块
包含各种评估指标的计算
"""

import numpy as np
import torch
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from collections import defaultdict
import re

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

from .config import EvaluationConfig
from .inference import Qwen3Inference

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = {}
        
        # 初始化评估器
        self._init_metrics()
    
    def _init_metrics(self):
        """初始化评估指标"""
        for metric in self.config.metrics:
            if metric == "rouge" and ROUGE_AVAILABLE:
                self.metrics["rouge"] = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
                )
            elif metric == "bleu" and BLEU_AVAILABLE:
                self.metrics["bleu"] = BLEU()
    
    def evaluate_predictions(
        self,
        predictions: List[str],
        references: List[str],
        inputs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """评估预测结果"""
        if len(predictions) != len(references):
            raise ValueError("预测结果和参考答案的数量不匹配")
        
        results = {}
        
        # 计算各种指标
        for metric_name in self.config.metrics:
            if metric_name == "bleu":
                results.update(self._compute_bleu(predictions, references))
            elif metric_name == "rouge":
                results.update(self._compute_rouge(predictions, references))
            elif metric_name == "exact_match":
                results["exact_match"] = self._compute_exact_match(predictions, references)
            elif metric_name == "length":
                results.update(self._compute_length_stats(predictions, references))
            elif metric_name == "perplexity":
                if inputs:
                    logger.warning("困惑度计算需要模型支持，当前跳过")
                else:
                    logger.warning("困惑度计算需要输入文本，当前跳过")
        
        # 计算平均值和其他统计信息
        results["total_samples"] = len(predictions)
        
        return results
    
    def _compute_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算BLEU分数"""
        if not BLEU_AVAILABLE:
            logger.warning("BLEU评估需要安装 sacrebleu: pip install sacrebleu")
            return {}
        
        try:
            # 处理中文分词
            if JIEBA_AVAILABLE:
                predictions = [' '.join(jieba.cut(pred)) for pred in predictions]
                references = [' '.join(jieba.cut(ref)) for ref in references]
            
            bleu = self.metrics["bleu"]
            scores = []
            
            for pred, ref in zip(predictions, references):
                score = bleu.sentence_score(pred, [ref])
                scores.append(score.score)
            
            return {
                "bleu": np.mean(scores),
                "bleu_std": np.std(scores)
            }
        except Exception as e:
            logger.error(f"BLEU计算错误: {e}")
            return {}
    
    def _compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE分数"""
        if not ROUGE_AVAILABLE:
            logger.warning("ROUGE评估需要安装 rouge-score: pip install rouge-score")
            return {}
        
        try:
            scorer = self.metrics["rouge"]
            rouge_scores = defaultdict(list)
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                for metric, score in scores.items():
                    rouge_scores[f"{metric}_precision"].append(score.precision)
                    rouge_scores[f"{metric}_recall"].append(score.recall)
                    rouge_scores[f"{metric}_f1"].append(score.fmeasure)
            
            # 计算平均值
            results = {}
            for metric, values in rouge_scores.items():
                results[metric] = np.mean(values)
                results[f"{metric}_std"] = np.std(values)
            
            return results
        except Exception as e:
            logger.error(f"ROUGE计算错误: {e}")
            return {}
    
    def _compute_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """计算精确匹配率"""
        matches = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip() == ref.strip())
        return matches / len(predictions)
    
    def _compute_length_stats(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算长度统计信息"""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        return {
            "avg_pred_length": np.mean(pred_lengths),
            "avg_ref_length": np.mean(ref_lengths),
            "pred_length_std": np.std(pred_lengths),
            "ref_length_std": np.std(ref_lengths),
            "length_ratio": np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
        }
    
    def evaluate_model_on_dataset(
        self,
        model: Qwen3Inference,
        test_data: List[Dict[str, str]],
        input_key: str = "input",
        output_key: str = "output",
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """在数据集上评估模型"""
        batch_size = batch_size or self.config.batch_size
        
        # 限制评估样本数量
        if self.config.max_eval_samples and len(test_data) > self.config.max_eval_samples:
            test_data = test_data[:self.config.max_eval_samples]
            logger.info(f"限制评估样本数量为: {self.config.max_eval_samples}")
        
        # 提取输入和参考答案
        inputs = [item[input_key] for item in test_data]
        references = [item[output_key] for item in test_data]
        
        # 生成预测
        logger.info(f"开始生成预测，共 {len(inputs)} 个样本")
        predictions = []
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_predictions = []
            
            for inp in batch_inputs:
                pred = model.generate(inp)
                batch_predictions.append(pred)
            
            predictions.extend(batch_predictions)
            logger.info(f"完成批次 {i//batch_size + 1}/{(len(inputs)-1)//batch_size + 1}")
        
        # 计算评估指标
        logger.info("计算评估指标...")
        results = self.evaluate_predictions(predictions, references, inputs)
        
        # 添加详细结果
        results["detailed_results"] = [
            {
                "input": inp,
                "reference": ref,
                "prediction": pred,
                "sample_id": i
            }
            for i, (inp, ref, pred) in enumerate(zip(inputs, references, predictions))
        ]
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], output_file: str):
        """保存评估结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 分别保存指标和详细结果
        metrics = {k: v for k, v in results.items() if k != "detailed_results"}
        detailed_results = results.get("detailed_results", [])
        
        # 保存指标
        metrics_file = output_path.parent / f"{output_path.stem}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 保存详细结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估指标已保存到: {metrics_file}")
        logger.info(f"详细结果已保存到: {output_file}")
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """生成评估报告"""
        report = "# 模型评估报告\n\n"
        
        # 基本信息
        report += f"## 基本信息\n"
        report += f"- 总样本数: {results.get('total_samples', 0)}\n"
        report += f"- 评估时间: {self._get_current_time()}\n\n"
        
        # 主要指标
        report += "## 主要指标\n"
        
        if "bleu" in results:
            report += f"- BLEU分数: {results['bleu']:.4f}\n"
        
        if "rouge1_f1" in results:
            report += f"- ROUGE-1 F1: {results['rouge1_f1']:.4f}\n"
            report += f"- ROUGE-2 F1: {results['rouge2_f1']:.4f}\n"
            report += f"- ROUGE-L F1: {results['rougeL_f1']:.4f}\n"
        
        if "exact_match" in results:
            report += f"- 精确匹配率: {results['exact_match']:.4f}\n"
        
        # 长度统计
        if "avg_pred_length" in results:
            report += "\n## 长度统计\n"
            report += f"- 平均预测长度: {results['avg_pred_length']:.2f}\n"
            report += f"- 平均参考长度: {results['avg_ref_length']:.2f}\n"
            report += f"- 长度比率: {results['length_ratio']:.2f}\n"
        
        # 样本分析
        if "detailed_results" in results:
            detailed = results["detailed_results"]
            if detailed:
                report += "\n## 样本分析\n"
                report += "### 前5个样本:\n"
                for i, sample in enumerate(detailed[:5]):
                    report += f"\n**样本 {i+1}:**\n"
                    report += f"- 输入: {sample['input'][:100]}...\n"
                    report += f"- 参考: {sample['reference'][:100]}...\n"
                    report += f"- 预测: {sample['prediction'][:100]}...\n"
        
        return report
    
    def _get_current_time(self) -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def compare_models(
        self,
        results_list: List[Dict[str, Any]],
        model_names: List[str]
    ) -> str:
        """比较多个模型的评估结果"""
        if len(results_list) != len(model_names):
            raise ValueError("结果数量和模型名称数量不匹配")
        
        report = "# 模型比较报告\n\n"
        
        # 创建比较表格
        metrics_to_compare = ["bleu", "rouge1_f1", "rouge2_f1", "rougeL_f1", "exact_match"]
        
        report += "## 主要指标比较\n\n"
        report += "| 模型 | "
        for metric in metrics_to_compare:
            report += f"{metric} | "
        report += "\n"
        
        report += "| --- | "
        for _ in metrics_to_compare:
            report += "--- | "
        report += "\n"
        
        for model_name, results in zip(model_names, results_list):
            report += f"| {model_name} | "
            for metric in metrics_to_compare:
                value = results.get(metric, 0)
                report += f"{value:.4f} | "
            report += "\n"
        
        # 找出最佳模型
        report += "\n## 最佳模型\n"
        for metric in metrics_to_compare:
            values = [results.get(metric, 0) for results in results_list]
            if values:
                best_idx = np.argmax(values)
                report += f"- {metric}: {model_names[best_idx]} ({values[best_idx]:.4f})\n"
        
        return report


class CustomMetrics:
    """自定义评估指标"""
    
    @staticmethod
    def html_structure_similarity(pred_html: str, ref_html: str) -> float:
        """计算HTML结构相似度（为html2figma场景设计）"""
        try:
            # 提取HTML标签
            pred_tags = re.findall(r'<(\w+)', pred_html.lower())
            ref_tags = re.findall(r'<(\w+)', ref_html.lower())
            
            # 计算标签集合的相似度
            pred_set = set(pred_tags)
            ref_set = set(ref_tags)
            
            if not ref_set:
                return 1.0 if not pred_set else 0.0
            
            intersection = len(pred_set & ref_set)
            union = len(pred_set | ref_set)
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def json_key_coverage(pred_json: str, ref_json: str) -> float:
        """计算JSON键覆盖率（为figma json场景设计）"""
        try:
            pred_data = json.loads(pred_json)
            ref_data = json.loads(ref_json)
            
            def extract_keys(obj, prefix=""):
                keys = set()
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        full_key = f"{prefix}.{k}" if prefix else k
                        keys.add(full_key)
                        keys.update(extract_keys(v, full_key))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        keys.update(extract_keys(item, f"{prefix}[{i}]"))
                return keys
            
            pred_keys = extract_keys(pred_data)
            ref_keys = extract_keys(ref_data)
            
            if not ref_keys:
                return 1.0 if not pred_keys else 0.0
            
            return len(pred_keys & ref_keys) / len(ref_keys)
        except (json.JSONDecodeError, Exception):
            return 0.0 