import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 기본 디렉토리 경로 정의
CACHE_DIR = "/data3/DB/LLM/Llama"
PROJECT_DIR = "/data3/jykim/PycharmProjects/HALoGEN"

# 모델 ID 정의
MODELS = {
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
}

# 작업 디렉토리 정의
PROMPTS_DIR = os.path.join(PROJECT_DIR, "prompts")
RESPONSES_DIR = os.path.join(PROJECT_DIR, "responses_test")

def load_prompts(task_name):
    """한 개의 프롬프트만 로드"""
    prompt_file = os.path.join(PROMPTS_DIR, f"prompts_{task_name}.csv")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"{prompt_file} 파일을 찾을 수 없습니다.")
    
    df = pd.read_csv(prompt_file)
    if 'prompt' not in df.columns:
        raise ValueError(f"{prompt_file} 파일에 'prompt' 열이 없습니다.")
    
    return [df['prompt'].iloc[0]]

def test_generate_responses():
    """테스트용 응답 생성"""
    if not os.path.exists(PROMPTS_DIR):
        raise FileNotFoundError(f"{PROMPTS_DIR} 디렉토리가 존재하지 않습니다.")

    os.makedirs(RESPONSES_DIR, exist_ok=True)
    
    task_name = "biographies"
    print(f"Testing task: {task_name}")
    
    task_dir = os.path.join(RESPONSES_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    try:
        prompts = load_prompts(task_name)
    except Exception as e:
        print(f"Error loading prompts: {str(e)}")
        return
    
    for model_name, model_id in MODELS.items():
        print(f"Testing model: {model_name}")
        print(f"Model ID: {model_id}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=CACHE_DIR,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # 결과를 저장할 리스트
            results = []
            for prompt in prompts:
                print(f"Testing prompt: {prompt[:50]}...")
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
            
            # DataFrame으로 변환하고 CSV로 저장
            df_results = pd.DataFrame(results)
            response_file = os.path.join(task_dir, f"{model_name}_test.csv")
            df_results.to_csv(response_file, index=False)
                
            print(f"Test response saved to {response_file}")
            
        except Exception as e:
            print(f"Error testing model {model_name}: {str(e)}")
            continue
        
        finally:
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test_generate_responses()