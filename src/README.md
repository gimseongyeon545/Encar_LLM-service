# 모델 Midm:2.0 Base 대신 Midm:2.0 Mini 사용시 변경 사항
- `"K-intelligence/Midm-2.0-Base-Instruct"` ➡️ `"K-intelligence/Midm-2.0-Mini-Instruct"`

</br>
</br>
</br>

- [1] **inference.py**
  - `MODEL_ID_DEFAULT = os.getenv("MIDM_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")` ➡️ `MODEL_ID_DEFAULT = os.getenv("MIDM_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`

</br>

- [2] **midm.py**
  - `DEFAULT_MODEL = os.getenv("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")` ➡️ `DEFAULT_MODEL = os.getenv("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`
  - `os.environ.setdefault("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")` ➡️ `os.environ.setdefault("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`
