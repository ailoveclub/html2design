#!/usr/bin/env python3
"""
Qwen3 模型训练脚本
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from qwen3_trainer import Qwen3Trainer
from qwen3_trainer.config import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Qwen3 模型训练")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        help="训练数据文件"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        help="验证数据文件"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="基础模型名称"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="批大小"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="学习率"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="使用LoRA微调"
    )
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="创建示例数据"
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config_manager = ConfigManager(args.config)
        
        # 命令行参数覆盖配置
        if args.output_dir:
            config_manager.training_config.output_dir = args.output_dir
        if args.train_file:
            config_manager.data_config.train_file = args.train_file
        if args.val_file:
            config_manager.data_config.val_file = args.val_file
        if args.model_name:
            config_manager.model_config.model_name = args.model_name
        if args.epochs:
            config_manager.training_config.num_train_epochs = args.epochs
        if args.batch_size:
            config_manager.training_config.per_device_train_batch_size = args.batch_size
        if args.learning_rate:
            config_manager.training_config.learning_rate = args.learning_rate
        
        # 创建示例数据（如果需要）
        if args.create_sample_data:
            from qwen3_trainer.data import DataLoader
            # 创建临时分词器用于生成示例数据
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config_manager.model_config.model_name,
                trust_remote_code=True
            )
            
            data_loader = DataLoader(config_manager.data_config, tokenizer)
            train_file, val_file = data_loader.create_sample_data()
            
            # 更新配置
            config_manager.data_config.train_file = train_file
            config_manager.data_config.val_file = val_file
            
            logger.info(f"已创建示例数据: {train_file}, {val_file}")
        
        # 创建训练器
        trainer = Qwen3Trainer(
            model_config=config_manager.model_config,
            training_config=config_manager.training_config,
            data_config=config_manager.data_config,
            use_lora=args.use_lora
        )
        
        # 开始训练
        logger.info("开始训练流程...")
        result = trainer.run_full_training()
        
        logger.info("训练完成！")
        logger.info(f"训练结果: {result.metrics}")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise


if __name__ == "__main__":
    main() 