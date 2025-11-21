# Encar 2025 AI 개발/기획 공모전
## Track2 AI 서비스 기획
## Contents
### 1. source code
- `midm.py`
- `inference.py`
- `streamlit_app.py`
### 2. 실행 방법
[1] 로컬 환경 (window 기준 명령어)
- 라이브러리: `requirements.txt` 참고
- commands
  ```
  python midm.py --prefetch
  ```
  ```
  streamlit run streamlit_app.py
  ```
[2] colab
- 기본 설치 commands
  ```
  !pip install bitsandbytes
  ```
  ```
  !pip install streamlit
  ```
- midm 모델 다운로드 commands
  ```
  !python midm.py --prefetch
  ```
- streamlit 실행 commands
  ```
  !streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 &>/content/streamlit.log &
  ```
  ```
  !curl -s -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
  ```
  ```
  !chmod +x cloudflared
  ```
  ```
  !./cloudflared tunnel --url http://localhost:8501 --no-autoupdate
  ```
### 3. 실행 화면
#### [1] 입력
- 매물: JSON 형식
- Persona & user_note: 일반 텍스트 형식
#### [2] 사용 모델: **Midm:2.0 Mini** 
- 좋은 환경에서 더 좋은 성능을 위해서는 기존 코드대로 Midm:2.0 Base 모델을 사용
- [코드 변경 부분]
  - [1] `inference.py`
    - `MODEL_ID_DEFAULT = os.getenv("MIDM_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`
  - [2] `midm.py`
    - `DEFAULT_MODEL = os.getenv("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`
#### [3] 실제 실행 결과
##### (1) Persona A1
