# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen2-Audio-7B"

# 第一次运行时会从 HuggingFace 下载到本地缓存
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

print("Qwen2-Audio 下载并加载完成 ✅")
