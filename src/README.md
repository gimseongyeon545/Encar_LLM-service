# 모델 Base 대신 Mini 사용시 변경 사항
- [1] `inference.py`
    - `MODEL_ID_DEFAULT = os.getenv("MIDM_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")` -> `MODEL_ID_DEFAULT = os.getenv("MIDM_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`
- [2] `midm.py`
  - `DEFAULT_MODEL = os.getenv("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")` -> `DEFAULT_MODEL = os.getenv("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`
  - 
