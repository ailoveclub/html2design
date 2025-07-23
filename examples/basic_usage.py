#!/usr/bin/env python3
"""
Qwen3 训练框架基本使用示例
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from qwen3_trainer import (
    Qwen3Trainer, 
    Qwen3Inference, 
    ModelEvaluator,
    ConfigManager
)
from qwen3_trainer.config import ModelConfig, InferenceConfig, EvaluationConfig


def example_1_basic_training():
    """示例1: 基本训练流程"""
    print("=" * 50)
    print("示例1: 基本训练流程")
    print("=" * 50)
    
    # 加载配置
    config_manager = ConfigManager("configs/default_config.yaml")
    
    # 创建训练器
    trainer = Qwen3Trainer(
        model_config=config_manager.model_config,
        training_config=config_manager.training_config,
        data_config=config_manager.data_config,
        use_lora=True
    )
    
    # 创建示例数据
    print("创建示例数据...")
    from qwen3_trainer.data import DataLoader
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        config_manager.model_config.model_name,
        trust_remote_code=True
    )
    
    data_loader = DataLoader(config_manager.data_config, tokenizer)
    train_file, val_file = data_loader.create_sample_data(num_samples=50)
    
    # 更新配置
    config_manager.data_config.train_file = train_file
    config_manager.data_config.val_file = val_file
    
    print(f"训练数据: {train_file}")
    print(f"验证数据: {val_file}")
    
    # 开始训练（这里只是演示，实际可能需要很长时间）
    print("开始训练（演示模式）...")
    print("注意: 完整训练需要较长时间，这里仅作演示")
    
    # 如果你想要实际运行训练，取消下面这行的注释
    # result = trainer.run_full_training()
    
    print("训练示例完成！")


def example_2_inference():
    """示例2: 模型推理"""
    print("\n" + "=" * 50)
    print("示例2: 模型推理")
    print("=" * 50)
    
    # 配置模型（使用预训练模型进行推理演示）
    model_config = ModelConfig()
    model_config.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    inference_config = InferenceConfig()
    inference_config.max_new_tokens = 100
    inference_config.temperature = 0.7
    
    # 创建推理引擎
    print("创建推理引擎...")
    inference_engine = Qwen3Inference(
        model_config=model_config,
        inference_config=inference_config,
        # 这里使用基础模型，如果有训练好的模型可以指定路径
        # model_path="./outputs",
        # is_lora_model=True
    )
    
    # 加载模型
    print("加载模型（可能需要几分钟）...")
    try:
        inference_engine.load_model()
        print("模型加载完成！")
        
        # 进行推理
        prompts = [
            "什么是人工智能？",
            "请解释一下深度学习的基本概念。",
            "如何学习编程？"
        ]
        
        print("\n开始推理:")
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. 输入: {prompt}")
            response = inference_engine.generate(prompt)
            print(f"   输出: {response}")
            
    except Exception as e:
        print(f"推理演示跳过（需要GPU或较大内存）: {e}")
    
    print("推理示例完成！")


def example_3_evaluation():
    """示例3: 模型评估"""
    print("\n" + "=" * 50)
    print("示例3: 模型评估")
    print("=" * 50)
    
    # 创建评估配置
    eval_config = EvaluationConfig()
    eval_config.metrics = ["bleu", "rouge", "exact_match", "length"]
    
    # 创建评估器
    evaluator = ModelEvaluator(eval_config)
    
    # 模拟评估数据
    predictions = [
        "人工智能是让机器模拟人类智能的技术。",
        "深度学习是机器学习的一个分支，使用神经网络进行学习。",
        "编程学习需要选择语言、练习基础语法、做项目实践。"
    ]
    
    references = [
        "人工智能是一门研究如何让机器模拟人类智能的学科。",
        "深度学习是机器学习的子集，通过多层神经网络进行学习。",
        "学习编程应该先选择编程语言，然后学习基础语法，最后通过项目实践。"
    ]
    
    # 进行评估
    print("进行评估...")
    try:
        results = evaluator.evaluate_predictions(predictions, references)
        
        print("评估结果:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        # 生成评估报告
        print("\n生成评估报告...")
        report = evaluator.generate_evaluation_report(results)
        print("报告预览:")
        print(report[:500] + "...")
        
    except Exception as e:
        print(f"评估演示（某些指标可能需要额外依赖）: {e}")
    
    print("评估示例完成！")


def example_4_config_management():
    """示例4: 配置管理"""
    print("\n" + "=" * 50)
    print("示例4: 配置管理")
    print("=" * 50)
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 修改配置
    config_manager.model_config.model_name = "Qwen/Qwen2.5-3B-Instruct"
    config_manager.training_config.num_train_epochs = 5
    config_manager.training_config.learning_rate = 1e-4
    
    print("当前配置:")
    print(f"模型名称: {config_manager.model_config.model_name}")
    print(f"训练轮数: {config_manager.training_config.num_train_epochs}")
    print(f"学习率: {config_manager.training_config.learning_rate}")
    
    # 保存配置
    config_file = "custom_config.yaml"
    config_manager.save_to_file(config_file)
    print(f"\n配置已保存到: {config_file}")
    
    # 重新加载配置
    new_config = ConfigManager(config_file)
    print(f"重新加载的模型名称: {new_config.model_config.model_name}")
    
    # 清理
    if os.path.exists(config_file):
        os.remove(config_file)
    
    print("配置管理示例完成！")


def example_5_html_to_figma_scenario():
    """示例5: HTML转Figma JSON场景演示"""
    print("\n" + "=" * 50)
    print("示例5: HTML转Figma JSON场景演示")
    print("=" * 50)
    
    # 模拟HTML转Figma JSON的数据
    html_inputs = [
        '<div class="container"><h1>标题</h1><p>段落文本</p></div>',
        '<button class="btn btn-primary">点击按钮</button>',
        '<img src="image.jpg" alt="图片" width="300" height="200">'
    ]
    
    figma_outputs = [
        '{"type": "FRAME", "children": [{"type": "TEXT", "characters": "标题"}, {"type": "TEXT", "characters": "段落文本"}]}',
        '{"type": "FRAME", "children": [{"type": "TEXT", "characters": "点击按钮", "fills": [{"type": "SOLID", "color": {"r": 0, "g": 0.5, "b": 1}}]}]}',
        '{"type": "RECTANGLE", "fills": [{"type": "IMAGE"}], "constraints": {"horizontal": "MIN", "vertical": "MIN"}}'
    ]
    
    print("HTML转Figma JSON数据示例:")
    for i, (html, figma) in enumerate(zip(html_inputs, figma_outputs), 1):
        print(f"\n{i}. HTML输入:")
        print(f"   {html}")
        print(f"   Figma JSON输出:")
        print(f"   {figma}")
    
    # 演示自定义评估指标
    print("\n使用自定义评估指标:")
    from qwen3_trainer.evaluator import CustomMetrics
    
    # HTML结构相似度
    pred_html = '<div><h1>标题</h1><p>文本</p></div>'
    ref_html = '<div class="container"><h1>标题</h1><p>段落文本</p></div>'
    
    similarity = CustomMetrics.html_structure_similarity(pred_html, ref_html)
    print(f"HTML结构相似度: {similarity:.4f}")
    
    # JSON键覆盖率
    pred_json = '{"type": "FRAME", "children": [{"type": "TEXT"}]}'
    ref_json = '{"type": "FRAME", "children": [{"type": "TEXT", "characters": "文本"}]}'
    
    coverage = CustomMetrics.json_key_coverage(pred_json, ref_json)
    print(f"JSON键覆盖率: {coverage:.4f}")
    
    print("HTML转Figma JSON场景演示完成！")


def main():
    """运行所有示例"""
    print("Qwen3 训练框架使用示例")
    print("注意: 某些示例可能需要GPU和较长时间运行")
    
    try:
        # 运行示例
        example_1_basic_training()
        example_2_inference()
        example_3_evaluation()
        example_4_config_management()
        example_5_html_to_figma_scenario()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        print("这可能是由于缺少某些依赖或硬件限制导致的")


if __name__ == "__main__":
    main() 