#!/usr/bin/env python3
"""
Qwen3 模型推理脚本
"""

import argparse
import logging
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src import Qwen3Inference, ModelConfig, InferenceConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Qwen3 模型推理")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./outputs",
        help="模型路径"
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
        "--prompt",
        type=str,
        help="输入提示"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="输入文件（每行一个提示）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="输出文件"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="最大生成长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度参数"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p参数"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互模式"
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="启动推理服务器"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器主机"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口"
    )
    
    args = parser.parse_args()
    
    try:
        # 配置模型
        model_config = ModelConfig()
        model_config.model_name = args.model_name
        
        inference_config = InferenceConfig()
        inference_config.max_new_tokens = args.max_new_tokens
        inference_config.temperature = args.temperature
        inference_config.top_p = args.top_p
        
        # 创建推理引擎
        inference_engine = Qwen3Inference(
            model_config=model_config,
            inference_config=inference_config,
            model_path=args.model_path if Path(args.model_path).exists() else None,
            is_lora_model=args.is_lora
        )
        
        # 加载模型
        logger.info("加载模型...")
        inference_engine.load_model()
        logger.info("模型加载完成")
        
        if args.start_server:
            # 启动推理服务器
            from src.inference import InferenceServer
            server = InferenceServer(inference_engine)
            logger.info(f"启动推理服务器: http://{args.host}:{args.port}")
            server.start_server(host=args.host, port=args.port)
            
        elif args.interactive:
            # 交互模式
            logger.info("进入交互模式，输入 'quit' 退出")
            history = []
            
            while True:
                try:
                    user_input = input("\n用户: ").strip()
                    if user_input.lower() in ['quit', 'exit', '退出']:
                        break
                    
                    if not user_input:
                        continue
                    
                    # 生成回复
                    response = inference_engine.chat(user_input, history)
                    print(f"助手: {response}")
                    
                    # 更新历史
                    history.append({
                        "user": user_input,
                        "assistant": response
                    })
                    
                    # 限制历史长度
                    if len(history) > 10:
                        history = history[-10:]
                        
                except KeyboardInterrupt:
                    print("\n再见！")
                    break
                except Exception as e:
                    print(f"生成错误: {e}")
        
        elif args.input_file:
            # 批量推理
            logger.info(f"从文件读取输入: {args.input_file}")
            
            with open(args.input_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            logger.info(f"开始批量推理，共 {len(prompts)} 个提示")
            results = []
            
            for i, prompt in enumerate(prompts):
                response = inference_engine.generate(prompt)
                result = {
                    "prompt": prompt,
                    "response": response,
                    "index": i
                }
                results.append(result)
                logger.info(f"完成 {i+1}/{len(prompts)}")
            
            # 保存结果
            output_file = args.output_file or "inference_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"结果已保存到: {output_file}")
        
        elif args.prompt:
            # 单次推理
            logger.info("开始推理...")
            response = inference_engine.generate(args.prompt)
            
            print(f"\n输入: {args.prompt}")
            print(f"输出: {response}")
            
            if args.output_file:
                result = {
                    "prompt": args.prompt,
                    "response": response
                }
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"结果已保存到: {args.output_file}")
        
        else:
            print("请指定 --prompt、--input-file、--interactive 或 --start-server 中的一个选项")
            
    except Exception as e:
        logger.error(f"推理过程中出错: {e}")
        raise


if __name__ == "__main__":
    main() 