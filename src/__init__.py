"""
Qwen3 训练框架
一个基于 Huggingface 的 Qwen3 文本生成模型训练框架
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import TrainingConfig, ModelConfig, DataConfig
from .data import DataLoader, DataProcessor
from .trainer import Qwen3Trainer
from .inference import Qwen3Inference
from .evaluator import ModelEvaluator

__all__ = [
    "TrainingConfig",
    "ModelConfig", 
    "DataConfig",
    "DataLoader",
    "DataProcessor",
    "Qwen3Trainer",
    "Qwen3Inference",
    "ModelEvaluator"
] 