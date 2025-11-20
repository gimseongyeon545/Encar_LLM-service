# Encar 2025 AI 개발/기획 공모전
## Track2 AI 서비스 기획
## Contents
### 1. source code
- `midm.py`
- `inference.py`
- `streamlit_app.py`
### 2. 실행 방법
[1] 일반 python 환경
```
python midm.py --prefetch
```
```
streamlit run streamlit_app.py
```
[2] colab
- 기본 설치
  ```
  !pip install bitsandbytes
  ```
  ```
  !pip install streamlit
  ```
- midm 모델 다운로드
  ```
  !python midm.py --prefetch
  ```
- streamlit 실행
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
#### [1] 입력 예시
- Json
#### [2] 실행 화면 예시
