import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 读取配置文件
with open("configs/default_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_cfg = config["model"]
model_name = model_cfg["model_name"]
cache_dir = model_cfg.get("cache_dir", None)
trust_remote_code = model_cfg.get("trust_remote_code", True)
torch_dtype = model_cfg.get("torch_dtype", "auto")
device_map = model_cfg.get("device_map", "auto")

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=trust_remote_code
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=trust_remote_code,
    torch_dtype=getattr(torch, torch_dtype) if torch_dtype != "auto" else None,
    device_map=device_map
)

# 对话循环
def build_prompt(history, user_input):
    # Qwen2.5 官方对话格式
    prompt = ""
    for turn in history:
        prompt += f"<|im_start|>user\n{turn['user']}<|im_end|>\n<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

if __name__ == "__main__":
    print("欢迎使用 Qwen2.5 对话推理示例，输入 exit 退出。")
    history = []
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("exit", "quit"): break
        prompt = build_prompt(history, user_input)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        # 截取 assistant 回复
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        print(f"Qwen: {response.strip()}")
        history.append({"user": user_input, "assistant": response.strip()}) 