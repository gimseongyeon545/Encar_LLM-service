# midm.py
# 목적: KT Mi:DM (HuggingFace) 로드/추론 캡슐화
# - 4bit(BitsAndBytes) 가능하면 사용, 아니면 자동 폴백
# - chat 템플릿 사용(인스트럭트 모델 안정적)
import os, warnings
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# ===== 기본 설정 =====
DEFAULT_MODEL = os.getenv("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Base-Instruct") # "K-intelligence/Midm-2.0-Mini-Instruct"

LOAD_IN_4BIT = os.getenv("MIDM_LOAD_IN_4BIT", "1") not in ("0", "false", "False")
MAX_NEW_TOKENS = int(os.getenv("MIDM_MAX_NEW_TOKENS", "768"))

# 선택: VRAM 제한 (필요 없으면 주석 처리)
_DEFAULT_MAX_MEMORY = {0: os.getenv("MIDM_MAX_MEMORY_GPU0", "10GiB"), "cpu": os.getenv("MIDM_MAX_MEMORY_CPU", "60GiB")}

_tok = None
_model = None
_model_name = None

def _ensure_loaded(name: str = DEFAULT_MODEL):
    """모델/토크나이저 1회 로드 후 캐시."""
    global _tok, _model, _model_name
    if _model is not None and _model_name == name:
        return

    # --- Tokenizer ---
    _tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    # pad 토큰 없으면 EOS로 대체 (generate 안정화)
    if getattr(_tok, "pad_token_id", None) is None and getattr(_tok, "eos_token_id", None) is not None:
        _tok.pad_token = _tok.eos_token

    # --- GenerationConfig (선택) ---
    try:
        _gen_cfg = GenerationConfig.from_pretrained(name)  # noqa: F841 (보관만)
    except Exception:
        pass

    # --- Model ---
    kwargs = {"trust_remote_code": True}

    # ✅ GPU 사용 가능 여부 + 강제 CPU 플래그
    force_cpu = os.getenv("MIDM_FORCE_CPU", "0") == "1"
    has_cuda = torch.cuda.is_available() and not force_cpu

    # 4bit 선호 → GPU 있을 때만 사용
    if LOAD_IN_4BIT and _HAS_BNB and has_cuda:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs.update(dict(
            quantization_config=bnb,
            device_map="auto",
            max_memory=_DEFAULT_MAX_MEMORY,   # GPU 0 + CPU 메모리 제한
        ))
    else:
        # 비양자화 로드: GPU 있으면 bfloat16/auto, 아니면 CPU float32
        if has_cuda:
            kwargs.update(dict(torch_dtype=torch.bfloat16, device_map="auto"))
        else:
            # ❗ GPU 없을 때는 device_map / max_memory 안 넘긴다 (CPU 단일 디바이스)
            kwargs.update(dict(torch_dtype=torch.float32))

    _model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    _model_name = name



def generate(messages: List[Dict[str, str]],
             max_new_tokens: int = MAX_NEW_TOKENS,
             do_sample: bool = False) -> str:
    """
    messages: [{"role":"user"/"assistant"/"system", "content": "..."}]
    return: 디코드된 텍스트(모델 출력 전체 문자열)
    """
    _ensure_loaded()

    # chat 템플릿 → input_ids
    input_ids = _tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(_model.device)

    with torch.no_grad():
        out = _model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=getattr(_tok, "eos_token_id", None),
            pad_token_id=getattr(_tok, "pad_token_id", None),
        )
    return _tok.decode(out[0], skip_special_tokens=True)


def generate_from_prompt(prompt: str,
                         max_new_tokens: int = MAX_NEW_TOKENS,
                         do_sample: bool = False) -> str:
    """단일 프롬프트로 호출하고 싶을 때."""
    msgs = [{"role": "user", "content": prompt}]
    return generate(msgs, max_new_tokens=max_new_tokens, do_sample=do_sample)


if __name__ == "__main__":
    import argparse
    from midm import _ensure_loaded, generate_from_prompt

    parser = argparse.ArgumentParser(description="KT Mi:DM helper")
    parser.add_argument("--prefetch", action="store_true",
                        help="모델/토크나이저 미리 다운로드만 수행하고 종료")
    parser.add_argument("--prompt", type=str, default=None,
                        help="단일 프롬프트로 한 번 추론해보기")
    parser.add_argument("--max_new_tokens", type=int, default=int(os.getenv("MIDM_MAX_NEW_TOKENS", "512")))
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")

    if args.prefetch:
        _ensure_loaded()
        print("✅ Mi:DM 모델/토크나이저 다운로드 및 로드 완료.")
    if args.prompt:
        out = generate_from_prompt(args.prompt, max_new_tokens=args.max_new_tokens, do_sample=False)
        print(out)
    if not args.prefetch and not args.prompt:
        print("사용법: python midm.py --prefetch  또는  python midm.py --prompt '안녕'")
