"""
训练模块
基于 Huggingface 的 Qwen3 模型训练器
"""

import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from .config import ModelConfig, TrainingConfig, DataConfig
from .data import DataLoader

logger = logging.getLogger(__name__)


class Qwen3Trainer:
    """Qwen3训练器"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.use_lora = use_lora
        self.lora_config = lora_config or self._get_default_lora_config()
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # 设置随机种子
        self._set_seed()
        
    def _set_seed(self):
        """设置随机种子"""
        import random
        import numpy as np
        
        random.seed(self.training_config.seed)
        np.random.seed(self.training_config.seed)
        torch.manual_seed(self.training_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.training_config.seed)
    
    def _get_default_lora_config(self) -> Dict[str, Any]:
        """获取默认的LoRA配置"""
        return {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info(f"加载模型: {self.model_config.model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            cache_dir=self.model_config.cache_dir,
            trust_remote_code=self.model_config.trust_remote_code,
            padding_side="right",
            padding=True,
            truncation=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            cache_dir=self.model_config.cache_dir,
            trust_remote_code=self.model_config.trust_remote_code,
            torch_dtype=getattr(torch, self.model_config.torch_dtype),
            device_map=self.model_config.device_map,
            #! 暂时不使用flash attention，因为模型不支持，换新的模型后可以试试
            # use_flash_attention_2=self.model_config.use_flash_attention
        )
        
        # 应用LoRA
        if self.use_lora:
            logger.info("应用LoRA配置")
            lora_config = LoraConfig(**self.lora_config)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        logger.info("模型和分词器加载完成")
    
    def prepare_datasets(self) -> Dict[str, Optional[Dataset]]:
        """准备数据集"""
        data_loader = DataLoader(self.data_config, self.tokenizer)
        
        datasets = {
            "train": data_loader.load_train_dataset(),
            "eval": data_loader.load_val_dataset(),
            "test": data_loader.load_test_dataset()
        }
        
        logger.info(f"训练集大小: {len(datasets['train']) if datasets['train'] else 0}")
        logger.info(f"验证集大小: {len(datasets['eval']) if datasets['eval'] else 0}")
        logger.info(f"测试集大小: {len(datasets['test']) if datasets['test'] else 0}")
        
        return datasets
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """设置训练器"""
        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            eval_steps=self.training_config.eval_steps if eval_dataset else None,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            # evaluation_strategy=self.training_config.evaluation_strategy if eval_dataset else "no",
            eval_strategy="steps",
            load_best_model_at_end=self.training_config.load_best_model_at_end and eval_dataset is not None,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            report_to=self.training_config.report_to,
            run_name=self.training_config.run_name,
            seed=self.training_config.seed,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            dataloader_pin_memory=self.training_config.dataloader_pin_memory,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )
        
        # 创建数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            # mlm=False,
            pad_to_multiple_of=8
        )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None
        )
        
        logger.info("训练器设置完成")
    
    def train(self):
        """开始训练"""
        if self.trainer is None:
            raise ValueError("训练器未初始化，请先调用 setup_trainer")
        
        logger.info("开始训练...")
        
        # 初始化wandb（如果配置了）
        if "wandb" in self.training_config.report_to:
            wandb.init(
                project="qwen3-finetuning",
                name=self.training_config.run_name,
                config=self.training_config.__dict__
            )
        
        # 开始训练
        train_result = self.trainer.train()
        
        # 保存模型
        self.trainer.save_model()
        self.trainer.save_state()
        
        # 保存训练指标
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info("训练完成")
        return train_result
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        """评估模型"""
        if self.trainer is None:
            raise ValueError("训练器未初始化")
        
        logger.info("开始评估...")
        
        if eval_dataset:
            # 使用指定的评估数据集
            eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        else:
            # 使用训练器中的评估数据集
            eval_result = self.trainer.evaluate()
        
        # 保存评估指标
        self.trainer.log_metrics("eval", eval_result)
        self.trainer.save_metrics("eval", eval_result)
        
        logger.info(f"评估结果: {eval_result}")
        return eval_result
    
    def save_model(self, output_dir: Optional[str] = None):
        """保存模型"""
        if output_dir is None:
            output_dir = self.training_config.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_lora and hasattr(self.model, 'save_pretrained'):
            # 保存LoRA权重
            self.model.save_pretrained(output_path)
            logger.info(f"LoRA模型已保存到: {output_path}")
        else:
            # 保存完整模型
            self.model.save_pretrained(output_path)
            logger.info(f"模型已保存到: {output_path}")
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"分词器已保存到: {output_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        logger.info(f"加载检查点: {checkpoint_path}")
        
        if self.trainer is None:
            raise ValueError("训练器未初始化")
        
        # 恢复训练状态
        self.trainer.train(resume_from_checkpoint=checkpoint_path)
    
    def run_full_training(self):
        """运行完整的训练流程"""
        # 1. 加载模型和分词器
        self.load_model_and_tokenizer()
        
        # 2. 准备数据集
        datasets = self.prepare_datasets()
        
        if datasets["train"] is None:
            raise ValueError("训练数据集为空")
        
        # 3. 设置训练器
        self.setup_trainer(datasets["train"], datasets["eval"])
        
        # 4. 开始训练
        train_result = self.train()
        
        # 5. 评估模型（如果有验证集）
        if datasets["eval"] is not None:
            eval_result = self.evaluate()
        
        # 6. 保存最终模型
        self.save_model()
        
        logger.info("完整训练流程完成")
        return train_result 