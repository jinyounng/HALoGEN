import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm 

BASE_DIR = "/data3/DB/LLM"
PROJECT_DIR = "/data3/jykim/PycharmProjects/HALoGEN"

MODELS = {
    "Llama-3.2-1B-Instruct": {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "cache_dir": os.path.join(BASE_DIR, "Llama")
    },
    "Llama-3.2-3B-Instruct": {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "cache_dir": os.path.join(BASE_DIR, "Llama")
    },
    "Qwen2.5-0.5B-Instruct": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "cache_dir": os.path.join(BASE_DIR, "Qwen2.5")
    },
    "Qwen2.5-1.5B-Instruct": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "cache_dir": os.path.join(BASE_DIR, "Qwen2.5")
    },
    "Qwen2.5-3B-Instruct": {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "cache_dir": os.path.join(BASE_DIR, "Qwen2.5")
    },
    "Qwen2.5-7B-Instruct": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "cache_dir": os.path.join(BASE_DIR, "Qwen2.5")
    },
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "cache_dir": os.path.join(BASE_DIR, "Deepseek")
    }
}

# 작업 디렉토리 정의
PROMPTS_DIR = os.path.join(PROJECT_DIR, "prompts")
RESPONSES_DIR = os.path.join(PROJECT_DIR, "responses")

def load_prompts(task_name):
    """프롬프트 파일을 로드합니다."""
    prompt_file = os.path.join(PROMPTS_DIR, f"prompts_{task_name}.csv")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"{prompt_file} 파일을 찾을 수 없습니다.")
    
    df = pd.read_csv(prompt_file)
    if 'prompt' not in df.columns:
        raise ValueError(f"{prompt_file} 파일에 'prompt' 열이 없습니다.")
    
    return df['prompt'].tolist()

def generate_and_save_responses():
    """각 작업과 모델에 대해 응답을 생성하고 저장합니다."""
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Responses directory: {RESPONSES_DIR}")
    
    if not os.path.exists(PROMPTS_DIR):
        raise FileNotFoundError(f"{PROMPTS_DIR} 디렉토리가 존재하지 않습니다.")

    os.makedirs(RESPONSES_DIR, exist_ok=True)
    
    # prompts 디렉토리의 모든 CSV 파일을 처리
    prompt_files = [f for f in os.listdir(PROMPTS_DIR) if f.startswith("prompts_") and f.endswith(".csv")]
    
    for prompt_file in prompt_files:
        task_name = prompt_file[len("prompts_"):-len(".csv")]
        print(f"\nProcessing task: {task_name}")
        
        task_dir = os.path.join(RESPONSES_DIR, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        print(f"Task directory: {task_dir}")
        
        try:
            prompts = load_prompts(task_name)
            print(f"Loaded {len(prompts)} prompts")
            
            for model_name, model_info in MODELS.items():
                print(f"\nProcessing model: {model_name}")
                print(f"Model ID: {model_info['id']}")
                print(f"Cache directory: {model_info['cache_dir']}")
                
                response_file = os.path.join(task_dir, f"{model_name}.csv")
                
                # 이미 처리된 모델은 건너뛰기
                if os.path.exists(response_file):
                    print(f"Skipping {model_name} - already processed")
                    continue
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_info['id'], 
                        cache_dir=model_info['cache_dir']
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_info['id'],
                        cache_dir=model_info['cache_dir'],
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    
                    results = []
                    # tqdm으로 progress bar 추가
                    for prompt in tqdm(prompts, desc=f"Processing {model_name}", ncols=100):
                        inputs = tokenizer(prompt, return_tensors="pt")
                        outputs = model.generate(
                           inputs["input_ids"],
                           max_length=100,
                           pad_token_id=tokenizer.eos_token_id
                       )
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        results.append({
                           'prompt': prompt,
                           'response': response,
                           'model': model_name
                        })
                    
                    # 결과를 DataFrame으로 변환하고 CSV로 저장
                    df_results = pd.DataFrame(results)
                    df_results.to_csv(response_file, index=False)
                    print(f"Successfully saved responses for {model_name}")
                    
                except Exception as e:
                    print(f"Error processing model {model_name}: {str(e)}")
                    continue
                
                finally:
                    if 'model' in locals():
                        del model
                    if 'tokenizer' in locals():
                        del tokenizer
                    torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing task {task_name}: {str(e)}")
            continue

if __name__ == "__main__":
    generate_and_save_responses()