# Encar 2025 AI 개발/기획 공모전
## Track2 AI 서비스 기획
### ➡️ Decision/Risk Coach: 검색을 넘어 확신으로, Persona 기반 중고차 의사결정·리스크 코치 AI

</br>
</br>

## Contents

### 📍1. source code
---
- `midm.py`
- `inference.py`
- `streamlit_app.py`

</br>
  
### 📍2. 실행 방법
---
[1] 로컬 환경 (window 기준 명령어)
- 라이브러리: `requirements.txt` 참고
  
- (1) 레포 다운로드
    ```
    git clone https://github.com/gimseongyeon545/Encar_LLM-service.git
    cd Encar_LLM-service
    ```
- (2) (선택) 가상환경
    ```
    python -m venv .encar
    ```
    ```
    .\.encar\Scripts\activate
    ```
- (3) 라이브러리 설치
    ```
    pip install -r requirements.txt
    ```
    ```
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
- (4) Midm 모델 다운로드 & Streamlit 앱 실행
    ```
    python src/midm.py --prefetch
    ```
    ```
    streamlit run src/streamlit_app.py
    ```

</br>
  
[2] colab
- gpu 설정: T4 GPU
  
- (0) 레포 다운로드 및 **src 폴더 내 3개 코드 colab 파일에 업로드**
    ```
    git clone https://github.com/gimseongyeon545/Encar_LLM-service.git
    ```
- (1) 기본 설치 commands
    ```
    !pip install bitsandbytes
    ```
    ```
    !pip install streamlit
    ```
- (2) midm 모델 다운로드 commands
    ```
    !python midm.py --prefetch
    ```
- (3) streamlit 실행 commands
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

</br>

### 📍3. 실행 화면
---
#### [1] 입력
- 매물: JSON 형식
- Persona & user_note: 일반 텍스트 형식
#### [2] (실행 화면을 위한) 사용 모델: **Midm:2.0 Mini** 
- **좋은 환경에서 더 좋은 성능을 위해서는 기존 코드대로 Midm:2.0 Base 모델을 사용**
- [코드 변경 부분]: `"K-intelligence/Midm-2.0-Base-Instruct"` ➡️ `"K-intelligence/Midm-2.0-Mini-Instruct"`
  - [1] **inference.py**
      - `MODEL_ID_DEFAULT = os.getenv("MIDM_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")` ➡️ `MODEL_ID_DEFAULT = os.getenv("MIDM_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`

  </br>
  
  - [2] **midm.py**
    - `DEFAULT_MODEL = os.getenv("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")` ➡️ `DEFAULT_MODEL = os.getenv("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`
    - `os.environ.setdefault("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")` ➡️ `os.environ.setdefault("TRANSFORMERS_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")`

#### [3] 실제 입력 및 실행 결과
##### (1) ✅ **Persona A1**
i. 입력
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

</br>

ii. 실행 결과 (결과 캡쳐)
    
  > <img width="432" height="1148" alt="Image" src="https://github.com/user-attachments/assets/726b98d8-024a-444e-a506-b0ec7641b176" />

</br>

##### (2) ✅ **Persona A2💸**
i. 입력
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
  - user_note: `장거리 운전이 필요해요. 1200만원 이하면 좋겠어요.`

</br>

ii. 실행 결과 (영상 및 결과 캡쳐)
  > "https://github.com/user-attachments/assets/6b97e279-fc91-40d3-88ad-fac2527e2923"
  > ![Image](https://github.com/user-attachments/assets/a2741ebb-e748-43d7-9245-e711784bbd6c)

</br>

  > <img width="432" height="1193" alt="Image" src="https://github.com/user-attachments/assets/282a3831-9eaf-48c2-a005-680d739e1488" />
    
</br>

##### (3) ✅ **Persona B**
i. 입력
  - JSON 입력
    ```
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
    }
    ```
  - Persona: `빨리 팔고 싶은 사람`

</br>

ii. 실행 결과 (결과 캡쳐)
  > <img width="432" height="939" alt="Image" src="https://github.com/user-attachments/assets/8a3136bd-26e9-4693-9052-00dbd7baeabf" />
