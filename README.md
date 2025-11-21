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
- JSON 입력
  ```
  [
    {
      "title": "쏘나타 DN8 2.0 가솔린 프리미엄",
      "year": 2021,
      "mileage_km": 48000,
      "price_krw": 18500000,
      "color": "금색",
      "accident_history": "앞펜더 단순교환 1회, 프레임 손상 없음",
      "usage_history": "렌트 이력 1년, 이후 개인 자가용 2년",
      "options": [
        "스마트크루즈",
        "차선이탈보조",
        "통풍시트",
        "후측방경보"
      ],
      "inspection": {
        "encar_inspection": "엔카진단+",
        "comments": "외관 경미한 스톤칩, 하부 부식 없음, 타이어 마모 40% 정도 남음"
      },
      "market_price_hint": "동급 평균 시세 대비 약간 낮은 편"
    },
    {
      "title": "K5 DL3 2.0 가솔린 노블레스",
      "year": 2020,
      "mileage_km": 62000,
      "price_krw": 17900000,
      "color": "핑크색",
      "accident_history": "무사고, 단순판금 도색 있음",
      "usage_history": "개인 출퇴근용 4년",
      "options": [
        "크루즈컨트롤",
        "차선이탈경고",
        "열선시트",
        "전방주차센서"
      ],
      "inspection": {
        "encar_inspection": "엔카진단",
        "comments": "외관 스크래치 일부, 하부 부식 없음, 타이어 마모 30% 정도 남음"
      },
      "market_price_hint": "동급 평균 시세와 비슷한 편"
    }
  ]
  ```
- Persona: `아기 있는 엄마`
- user_note: `장거리 운전이 필요해요.`
- 실행 결과 (영상 및 결과 캡쳐)
  https://github.com/gimseongyeon545/Encar_LLM-service/issues/1#issue-3651384931
  ![Image](https://github.com/user-attachments/assets/a2741ebb-e748-43d7-9245-e711784bbd6c)
  
