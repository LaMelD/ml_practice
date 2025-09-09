import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


LOCAL_MODEL_PATH = "../ai_models/gemma-3-270m"
DTYPE = torch.float32
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

# ==== 생성 설정 ====
gen_cfg = GenerationConfig(
    max_new_tokens=256,       # 너무 길면 느려질 수 있으니 128~256 권장
    temperature=0.7,
    top_p=0.9,
    do_sample=True,           # 다양성 확보
    repetition_penalty=1.05,  # 중복 억제 살짝
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,  # pad 경고 방지
)

user_input = "안녕하세요. 당신은 누구입니까?"
inputs = build_inputs(user_input).to(device)

# ==== 생성 ====
print("🚀 Generating...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        do_sample=gen_cfg.do_sample,
        repetition_penalty=gen_cfg.repetition_penalty,
        eos_token_id=gen_cfg.eos_token_id,
        pad_token_id=gen_cfg.pad_token_id,
    )

# 디코딩 (프롬프트 부분 제외)
gen_only = output_ids[0][inputs["input_ids"].shape[-1]:]
text = tokenizer.decode(gen_only, skip_special_tokens=True)

print("\n===== MODEL OUTPUT =====")
print(text)
print("========================")