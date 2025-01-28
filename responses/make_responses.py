import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = "/data3/DB/LLM"
DIRS = {
    "LLAMA": os.path.join(BASE_DIR, "Llama"),
    "QWEN": os.path.join(BASE_DIR, "Qwen/Qwen"),
    "DEEPSEEK": os.path.join(BASE_DIR, "Deepseek/deepseek-ai")
}

PROMPTS_DIR = "prompts"  # 프롬프트 파일들이 저장된 디렉토리
RESPONSES_DIR = "responses"  # 응답을 저장할 디렉토리

MODEL_NAMES = {
    "LLAMA": [
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct"
    ],
    "QWEN": [
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct"
    ],
    "DEEPSEEK": [
        "DeepSeek-R1-Distill-Qwen-1.5B"
    ]
}

MODELS = {
    model_name: os.path.join(DIRS[model_type], model_name)
    for model_type in MODEL_NAMES
    for model_name in MODEL_NAMES[model_type]
}

def load_prompts(task_name):
    prompt_file = os.path.join(PROMPTS_DIR, f"prompt_{task_name}.csv")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"{prompt_file} 파일을 찾을 수 없습니다.")
    
    df = pd.read_csv(prompt_file)
    if 'prompts' not in df.columns:
        raise ValueError(f"{prompt_file} 파일에 'prompts' 열이 없습니다.")
    
    return df['prompts'].tolist()

def generate_and_save_responses():
    if not os.path.exists(PROMPTS_DIR):
        raise FileNotFoundError(f"{PROMPTS_DIR} 디렉토리가 존재하지 않습니다.")

    os.makedirs(RESPONSES_DIR, exist_ok=True)
    
    # 프롬프트 파일들을 처리
    for prompt_file in os.listdir(PROMPTS_DIR):
        if prompt_file.startswith("prompt_") and prompt_file.endswith(".csv"):
            task_name = prompt_file[len("prompt_"):-len(".csv")]
            print(f"Processing task: {task_name}")
            
            # 태스크별 디렉토리 생성
            task_dir = os.path.join(RESPONSES_DIR, task_name)
            os.makedirs(task_dir, exist_ok=True)
            
            prompts = load_prompts(task_name)

            for model_name, model_path in MODELS.items():
                print(f"  Using model: {model_name}")
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    
                    responses = []
                    for i, prompt in enumerate(prompts):
                        print(f"    Processing prompt {i+1}/{len(prompts)}")
                        inputs = tokenizer(prompt, return_tensors="pt")
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_length=100,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        responses.append({
                            "prompt": prompt,
                            "response": response
                        })
                    
                    # 응답 저장
                    response_file = os.path.join(task_dir, f"{model_name}.json")
                    with open(response_file, "w", encoding="utf-8") as f:
                        json.dump(responses, f, ensure_ascii=False, indent=4)
                        
                    print(f"    Responses saved to {response_file}")
                    
                except Exception as e:
                    print(f"Error processing model {model_name}: {str(e)}")
                    continue
                
                finally:
                    # 메모리 해제
                    if 'model' in locals():
                        del model
                    if 'tokenizer' in locals():
                        del tokenizer
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    generate_and_save_responses()