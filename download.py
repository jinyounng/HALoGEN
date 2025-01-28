from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/data3/DB/LLM/Llama"
cache_dir2 = "/data3/DB/LLM/Qwen"
cache_dir3 = "/data3/DB/LLM/Deepseek"

# tokenizer = AutoTokenizer.from_pretrained(
#     "meta-llama/Llama-3.3-70B-Instruct",
#     device_map = "auto",
#     cache_dir=cache_dir
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.3-70B-Instruct",
#     cache_dir=cache_dir
# )
# LLAMA
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    cache_dir=cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    cache_dir=cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    cache_dir=cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    cache_dir=cache_dir
)

#Qwen Models 

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir=cache_dir2
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir=cache_dir2
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    cache_dir=cache_dir2
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    cache_dir=cache_dir2
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir=cache_dir2
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir=cache_dir2
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    cache_dir=cache_dir2
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    cache_dir=cache_dir2
)

# Deepseek models
tokenizer1 = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    cache_dir=cache_dir3
)
model1 = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    cache_dir=cache_dir3
)