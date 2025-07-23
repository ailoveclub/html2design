#!/usr/bin/env python3
"""
Qwen3 模型评估脚本
"""

import argparse
import logging
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from qwen3_trainer import Qwen3Inference, ModelEvaluator, ModelConfig, InferenceConfig, EvaluationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Qwen3 模型评估")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="测试数据文件"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="基础模型名称"
    )
    parser.add_argument(
        "--is-lora",
        action="store_true",
        help="是否为LoRA模型"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="输出目录"
    )
    parser.add_argument(
        "--input-column",
        type=str,
        default="input",
        help="输入列名"
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default="output",
        help="输出列名"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["bleu", "rouge", "exact_match", "length"],
        help="评估指标"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批大小"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="最大评估样本数"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="生成评估报告"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建输出目录
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 配置模型
        model_config = ModelConfig()
        model_config.model_name = args.model_name
        
        inference_config = InferenceConfig()
        
        evaluation_config = EvaluationConfig()
        evaluation_config.metrics = args.metrics
        evaluation_config.batch_size = args.batch_size
        evaluation_config.max_eval_samples = args.max_samples
        
        # 创建推理引擎
        inference_engine = Qwen3Inference(
            model_config=model_config,
            inference_config=inference_config,
            model_path=args.model_path,
            is_lora_model=args.is_lora
        )
        
        # 加载模型
        logger.info("加载模型...")
        inference_engine.load_model()
        logger.info("模型加载完成")
        
        # 加载测试数据
        logger.info(f"加载测试数据: {args.test_file}")
        test_data = []
        
        if args.test_file.endswith('.jsonl'):
            with open(args.test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line))
        elif args.test_file.endswith('.json'):
            with open(args.test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    test_data = data
                else:
                    test_data = [data]
        else:
            raise ValueError("不支持的文件格式，请使用 .json 或 .jsonl")
        
        logger.info(f"加载了 {len(test_data)} 个测试样本")
        
        # 创建评估器
        evaluator = ModelEvaluator(evaluation_config)
        
        # 进行评估
        logger.info("开始评估...")
        results = evaluator.evaluate_model_on_dataset(
            model=inference_engine,
            test_data=test_data,
            input_key=args.input_column,
            output_key=args.output_column
        )
        
        # 保存评估结果
        results_file = output_path / "evaluation_results.json"
        evaluator.save_evaluation_results(results, str(results_file))
        
        # 打印主要指标
        logger.info("评估完成！主要指标:")
        for metric, value in results.items():
            if metric != "detailed_results" and isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        # 生成评估报告
        if args.generate_report:
            logger.info("生成评估报告...")
            report = evaluator.generate_evaluation_report(results)
            
            report_file = output_path / "evaluation_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"评估报告已保存到: {report_file}")
        
        logger.info(f"所有结果已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        raise


if __name__ == "__main__":
    main() 