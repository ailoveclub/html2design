"""
推理模块
支持模型加载和文本生成
"""

import torch
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel
import json

from .config import ModelConfig, InferenceConfig

logger = logging.getLogger(__name__)


class Qwen3Inference:
    """Qwen3推理器"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        model_path: Optional[str] = None,
        is_lora_model: bool = False
    ):
        self.model_config = model_config
        self.inference_config = inference_config
        self.model_path = model_path or model_config.model_path
        self.is_lora_model = is_lora_model
        
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
    def load_model(self):
        """加载模型和分词器"""
        logger.info(f"加载推理模型: {self.model_path or self.model_config.model_name}")
        
        # 确定模型路径
        model_name_or_path = self.model_path or self.model_config.model_name
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=self.model_config.cache_dir,
            trust_remote_code=self.model_config.trust_remote_code,
            padding_side="left"  # 推理时使用左侧padding
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        if self.is_lora_model and self.model_path:
            # 如果是LoRA模型，先加载基础模型，再加载LoRA权重
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code,
                torch_dtype=getattr(torch, self.model_config.torch_dtype),
                device_map=self.model_config.device_map,
                use_flash_attention_2=self.model_config.use_flash_attention
            )
            
            # 加载LoRA权重
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.merge_and_unload()  # 合并LoRA权重
            logger.info("LoRA权重已合并到基础模型")
        else:
            # 直接加载完整模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code,
                torch_dtype=getattr(torch, self.model_config.torch_dtype),
                device_map=self.model_config.device_map,
                use_flash_attention_2=self.model_config.use_flash_attention
            )
        
        # 设置生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=self.inference_config.max_new_tokens,
            temperature=self.inference_config.temperature,
            top_p=self.inference_config.top_p,
            top_k=self.inference_config.top_k,
            do_sample=self.inference_config.do_sample,
            repetition_penalty=self.inference_config.repetition_penalty,
            pad_token_id=self.inference_config.pad_token_id or self.tokenizer.pad_token_id,
            eos_token_id=self.inference_config.eos_token_id or self.tokenizer.eos_token_id
        )
        
        # 设置为评估模式
        self.model.eval()
        
        logger.info("推理模型加载完成")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> str:
        """生成文本"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用 load_model()")
        
        # 格式化输入
        formatted_prompt = self._format_prompt(prompt)
        
        # 分词
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.max_sequence_length
        )
        
        # 移动到设备
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 创建生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens or self.inference_config.max_new_tokens,
            temperature=temperature or self.inference_config.temperature,
            top_p=top_p or self.inference_config.top_p,
            top_k=top_k or self.inference_config.top_k,
            do_sample=do_sample if do_sample is not None else self.inference_config.do_sample,
            repetition_penalty=self.inference_config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # 解码
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """批量生成文本"""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                result = self.generate(prompt, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
            logger.info(f"完成批次 {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
        
        return results
    
    def _format_prompt(self, prompt: str) -> str:
        """格式化输入提示"""
        # 使用Qwen的对话格式
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """对话模式"""
        if history is None:
            history = []
        
        # 构建对话历史
        conversation = ""
        for turn in history:
            conversation += f"<|im_start|>user\n{turn['user']}<|im_end|>\n"
            conversation += f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"
        
        # 添加当前消息
        conversation += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        
        # 生成回复
        response = self.generate(conversation)
        
        return response
    
    def save_generation_config(self, output_path: str):
        """保存生成配置"""
        config_dict = {
            "max_new_tokens": self.inference_config.max_new_tokens,
            "temperature": self.inference_config.temperature,
            "top_p": self.inference_config.top_p,
            "top_k": self.inference_config.top_k,
            "do_sample": self.inference_config.do_sample,
            "repetition_penalty": self.inference_config.repetition_penalty,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"生成配置已保存到: {output_path}")
    
    def evaluate_on_dataset(
        self,
        test_data: List[Dict[str, str]],
        input_key: str = "input",
        output_key: str = "output",
        save_results: bool = True,
        output_file: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """在数据集上评估模型"""
        results = []
        
        logger.info(f"开始评估，共 {len(test_data)} 个样本")
        
        for i, item in enumerate(test_data):
            input_text = item[input_key]
            expected_output = item.get(output_key, "")
            
            # 生成预测
            predicted_output = self.generate(input_text)
            
            result = {
                "input": input_text,
                "expected": expected_output,
                "predicted": predicted_output,
                "sample_id": i
            }
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"已完成 {i + 1}/{len(test_data)} 个样本")
        
        # 保存结果
        if save_results:
            if output_file is None:
                output_file = "evaluation_results.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"评估结果已保存到: {output_file}")
        
        return results


class InferenceServer:
    """推理服务器（简单版本）"""
    
    def __init__(self, inference_engine: Qwen3Inference):
        self.inference_engine = inference_engine
        
    def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """启动推理服务器"""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            import uvicorn
            
            app = FastAPI(title="Qwen3 推理服务", version="1.0.0")
            
            class GenerateRequest(BaseModel):
                prompt: str
                max_new_tokens: Optional[int] = None
                temperature: Optional[float] = None
                top_p: Optional[float] = None
                top_k: Optional[int] = None
                do_sample: Optional[bool] = None
            
            class GenerateResponse(BaseModel):
                generated_text: str
                prompt: str
            
            @app.post("/generate", response_model=GenerateResponse)
            async def generate(request: GenerateRequest):
                try:
                    generated_text = self.inference_engine.generate(
                        prompt=request.prompt,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        do_sample=request.do_sample
                    )
                    return GenerateResponse(
                        generated_text=generated_text,
                        prompt=request.prompt
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.get("/health")
            async def health_check():
                return {"status": "healthy"}
            
            logger.info(f"启动推理服务器: http://{host}:{port}")
            uvicorn.run(app, host=host, port=port)
            
        except ImportError:
            logger.error("启动服务器需要安装 fastapi 和 uvicorn: pip install fastapi uvicorn")
            raise 