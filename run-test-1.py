import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType

LOCAL_MODEL_PATH = "../ai_models/gemma-3-270m"
DTYPE = torch.bfloat16
MODEL_OPTION = {"use_safetensors": True}
ADAPTER_FLAG = False
ADAPTER_PATH = ""

device = None
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available; using CPU")

# ==== 토크나이저 로드 ====
def get_tokenizer(model_path):
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left",  # 배치 추론 대비 안전
        use_safetensors=True,
    )
    if tokenizer.pad_token is None:
        print("⚠️ pad_token이 없어서 eos_token으로 설정합니다.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    return tokenizer

def get_model(model_path, dtype, option):
    print("🔄 Loading model...")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=option["use_safetensors"],
    )

def set_model_to_device(model, device):
    print("🔄 Moving model to device...")
    model.to(device)
    model.eval()
    return model

tokenizer = get_tokenizer(LOCAL_MODEL_PATH)
model = get_model(LOCAL_MODEL_PATH, DTYPE, MODEL_OPTION)
model = set_model_to_device(model, device)

print("load done.")

# ==== 프롬프트 구성: chat 템플릿 자동 감지 ====
def build_inputs(user_text: str):
    # tokenizer가 chat 템플릿을 제공하면 그걸 사용 (instruct 모델에 유리)
    has_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
    if has_chat:
        messages = [{"role": "user", "content": user_text}]
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
    else:
        # 베이스 모델일 경우 단순 프롬프트
        enc = tokenizer(
            user_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
    return enc

# 옵션에 따른 추론 테스트
# 퍼포먼스 향상 테스트
print("===== generate =====")
# gen_cfg = GenerationConfig(
#     max_new_tokens=256,       # 너무 길면 느려질 수 있으니 128~256 권장
#     temperature=None,      # 다양성 확보
#     top_p=None,
#     do_sample=False,           # 다양성 확보
#     repetition_penalty=1.05,  # 중복 억제 살짝
#     eos_token_id=tokenizer.eos_token_id,
#     pad_token_id=tokenizer.eos_token_id,  # pad 경고 방지
# )

system_prompt = """너는 인공지능 대화 모델이야.
대답은 간단하고 사실적으로 해야 해.
자신에 대해 물으면 "저는 Gemma 3 270M 모델입니다."라고만 답해."""
user_input = "피보나치 수열에 대해 설명해주세요."

format_user_input = f"{system_prompt}\n사용자: {user_input}\n답변:"
inputs = tokenizer(format_user_input, return_tensors="pt")
device = model.device
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    gen_cfg = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,           # 샘플링 끔
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        num_beams=3,               # 빔서치
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # gen_cfg.to_dict()를 json pretty print로 확인해보세요.
    json_pretty = gen_cfg.to_dict()
    import json
    print(json.dumps(json_pretty, indent=2, ensure_ascii=False))

    outputs = model.generate(
        **inputs,
        **gen_cfg.to_dict()
    )

# 디코딩 (프롬프트 부분 제외)
gen_only = outputs[0][inputs["input_ids"].shape[-1]:]
text = tokenizer.decode(gen_only, skip_special_tokens=True)
raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n===== MODEL RAW OUTPUT =====")
print(raw)
print("========================")

print("\n===== MODEL OUTPUT =====")
print(text)
print("========================")