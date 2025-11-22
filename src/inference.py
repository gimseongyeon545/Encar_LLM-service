# inference.py
# 역할 기반(모드별) 엔카 코파일럿 inference 모듈
# - 단일 매물: generate_view(...)
# - 여러 매물 비교: generate_multi_view(...)

from __future__ import annotations

import os
import json
import textwrap
from dataclasses import dataclass
from typing import Dict, Any, List, Literal, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


Mode = Literal["buy", "sell"]


# ==============================
# 1. 페르소나 정의
# ==============================

@dataclass
class Persona:
    id: str          # internal id
    label: str       # 한글 라벨
    mode: Mode       # "buy" or "sell"
    description: str # 상황/우선순위 설명


BUY_PERSONAS: Dict[str, Persona] = {
    "first_car_student": Persona(
        id="first_car_student",
        label="첫 차 사는 대학생",
        mode="buy",
        description=(
            "운전 경력은 많지 않고, 첫 차를 구매하는 대학생이다. "
            "예산이 넉넉하지 않고, 유지비와 보험료, 주차 난이도가 중요하다. "
            "안전성과 기본적인 편의 기능은 중요하지만, 고급 옵션이나 출력은 덜 중요하다."
        ),
    ),
    "beginner_driver": Persona(
        id="beginner_driver",
        label="초보 운전자",
        mode="buy",
        description=(
            "운전 경력이 짧아 차 폭/길이, 시야, 주차 편의성이 중요하다. "
            "큰 사고 이력이나 수리비가 많이 나올 수 있는 차량은 피하고 싶다. "
            "운전이 편하고 실수해도 크게 위험하지 않은 차를 선호한다."
        ),
    ),
    "family_second_car": Persona(
        id="family_second_car",
        label="가족용 세컨카(30대)",
        mode="buy",
        description=(
            "아이를 포함한 가족이 함께 타는 세컨카를 찾는 30대 가장/부부다. "
            "뒷좌석 공간, 유아용 카시트 장착(ISOFIX), 트렁크 적재 공간, 승차감, 안전장비가 매우 중요하다. "
            "고속 주행 성능보다 편안함과 안전, 유지비의 합리성을 중시한다."
        ),
    ),
    "sales_commute": Persona(
        id="sales_commute",
        label="영업/출퇴근용",
        mode="buy",
        description=(
            "하루 평균 주행거리가 길고, 고속도로/시외도로를 자주 타는 직장인 혹은 영업사원이다. "
            "연비, 내구성, 고속 주행 안정성, 정비 편의성이 매우 중요하다. "
            "실내 소음/진동도 장거리 피로도에 영향을 준다."
        ),
    ),
    "enthusiast": Persona(
        id="enthusiast",
        label="차 좀 아는 사람(고수 모드)",
        mode="buy",
        description=(
            "차량에 대한 지식이 어느 정도 있고, 옵션/트림/사고 이력/감가 등을 세밀하게 본다. "
            "단순교환과 구조부 손상, 전손/침수, 수리 이력 차이를 구분할 줄 알고, "
            "시세 대비 메리트가 있는지, 향후 되팔 때 감가까지 고려한다."
        ),
    ),
}

SELL_PERSONAS: Dict[str, Persona] = {
    "sell_fast": Persona(
        id="sell_fast",
        label="빨리 팔고 싶은 사람",
        mode="sell",
        description=(
            "최대한 빠르게 차량을 처분하는 것이 1순위인 판매자다. "
            "약간의 금전적 손해는 감수할 수 있지만, "
            "복잡한 협상/네고/직거래 과정은 피하고 싶어 한다."
        ),
    ),
    "sell_best_price": Persona(
        id="sell_best_price",
        label="제값 이상 받고 싶은 사람",
        mode="sell",
        description=(
            "시간이 조금 더 걸리더라도, 차량 상태/옵션을 잘 어필해서 "
            "가능한 한 높은 가격으로 판매하고 싶은 판매자다. "
            "사진과 설명을 공들여 쓰는 것은 괜찮지만, 과장/허위는 피하고 싶다."
        ),
    ),
    "sell_easy": Persona(
        id="sell_easy",
        label="귀찮은 거 최소화",
        mode="sell",
        description=(
            "서류/탁송/네고 등 복잡한 과정을 최소화하고 싶다. "
            "가격은 어느 정도만 합리적이면 되고, 내 시간을 많이 쓰고 싶지 않은 판매자다."
        ),
    ),
    "sell_safe": Persona(
        id="sell_safe",
        label="안전/분쟁 최소화 우선",
        mode="sell",
        description=(
            "나중에 분쟁이 생기지 않도록 사실 기반으로 솔직하게 판매하고 싶다. "
            "사고 이력/수리 이력을 숨기고 싶지 않고, "
            "계약 조건과 책임 범위를 명확히 하고 싶어 한다."
        ),
    ),
}


def get_persona(persona_id: str, mode: Mode) -> Persona:
    """persona_id + mode 에 맞는 Persona 객체를 반환."""
    table = BUY_PERSONAS if mode == "buy" else SELL_PERSONAS
    if persona_id not in table:
        raise ValueError(f"Unknown persona_id for mode='{mode}': {persona_id}")
    return table[persona_id]


# ==============================
# 2. 프롬프트 빌더
# ==============================

# ==============================
# (추가) 멀티 매물용 차량 데이터 압축
# ==============================
def _shrink_vehicle_for_multi(v: Dict[str, Any]) -> Dict[str, Any]:
    """
    멀티 매물 비교 시, 컨텍스트 길이를 줄이기 위해
    꼭 필요한 필드만 남기고 긴 텍스트는 잘라서 사용.
    """
    # 1) 우선 키를 줄이자 (필요한 것만)
    KEYS_KEEP = [
        "title",
        "year",
        "mileage_km",
        "price_krw",
        "color",
        "accident_history",
        "usage_history",
        "market_price_hint",
        "options",
    ]
    out: Dict[str, Any] = {}
    for k in KEYS_KEEP:
        if k in v:
            out[k] = v[k]

    # 2) options 개수 제한
    if isinstance(out.get("options"), list) and len(out["options"]) > 6:
        out["options"] = out["options"][:6]

    # 3) 문자열은 너무 길면 자르기
    for k, val in list(out.items()):
        if isinstance(val, str) and len(val) > 120:
            out[k] = val[:120] + "..."

    # 4) 중첩 dict(inspection 등)는 아예 빼버리거나 아주 요약만 남기고 싶으면 여기서 처리
    # ex)
    # insp = v.get("inspection")
    # if isinstance(insp, dict):
    #     out["inspection_summary"] = insp.get("encar_inspection", "")

    return out


import re

def _has_budget(user_note: Optional[str]) -> bool:
    if not user_note:
        return False
    text = user_note.replace(" ", "")
    # 예시 패턴: "1200만원이하", "1500까지", "예산은 1000 정도"
    patterns = [
        r"\d+\s*만원\s*(이하|까지|정도)",
        r"예산\s*[:은]\s*\d+\s*만",
    ]
    return any(re.search(p, text) for p in patterns)




def build_prompt(
    vehicle_data: Dict[str, Any] | List[Dict[str, Any]],
    persona: Persona,
    user_note: Optional[str] = None,
    
) -> str:
    """
    단일/다중 매물 모두 지원하는 공통 프롬프트 빌더.
    - generate_view 에서는 단일 dict 로 사용
    - generate_multi_view 에서는 build_multi_prompt 를 쓰므로,
      여기의 list 분기는 주로 테스트/호환용.
    """
    has_user_note = bool(user_note and user_note.strip())
    has_budget = _has_budget(user_note)  # 🔹 예산 유무
    is_multi = isinstance(vehicle_data, list)

    # ---------- A. 여러 매물 비교용 (호환용) ----------
    if is_multi:
        vehicles_json = json.dumps(vehicle_data, ensure_ascii=False, indent=2)

        base_instruction = textwrap.dedent("""
        당신은 중고차를 고르는 사람에게 조언해주는 도우미입니다.

        아래 [persona]는 이 매물을 보고 있는 사람의 상황/목적/성향을 설명합니다.
        아래 [vehicles]는 여러 대의 매물에 대한 구조화된 정보 목록입니다.

        이 사람은 이 중에서 "나에게 더 잘 맞는 차"를 고르고 싶어 합니다.

        원칙:
        - 항상 persona의 관점에서 생각하고 말하세요.
        - 자동차/보험/정비 관련 전문 용어를 남발하지 말고, 필요하면 짧게 풀어서 설명하세요.
        - 각 매물의 절대적인 좋고 나쁨이 아니라, persona에게 "상대적으로" 더 잘 맞는지 판단하세요.
        - vehicle_data에 없는 정보(보험료, 세금, 실제 연비 등)는 추측해서 단정하지 말고,
          "이 JSON만으로는 정확히 알 수 없다"고 분명하게 말하세요.
          다만 일반적인 경향을 말할 때는 "보통 ~인 경우가 많다" 수준으로만 설명하세요.

        출력 규칙(중요):
        - 반드시 하나의 JSON 객체만 출력하세요.
        - "요약", "장점" 같은 제목/설명 문장을 JSON 바깥에 쓰지 마세요.
        - JSON 코드 블록이나 ```json 같은 래핑 없이, 순수 JSON만 출력하세요.

        JSON 스키마 예시는 아래와 같습니다. key 이름과 구조를 그대로 따르세요.

        {
          "mode": "buy",
          "persona_id": "...",
          "persona_label": "...",

          "summary": "여러 매물을 persona 관점에서 한 문단 정도로 요약",
          "highlights": [
            "여러 매물 비교에서 특히 중요한 핵심 포인트를 3~5개 bullet 로 정리 (persona 기준)"
          ],
          "pros": [
            "전체적으로 persona 입장에서의 장점 (2~5개)"
          ],
          "cons": [
            "전체적으로 persona 입장에서의 주의사항/단점 (2~5개)"
          ],
          "risk_level": "low | medium | high",
          "checklist": [
            "시승/상담 시 공통으로 꼭 확인해야 할 항목 (3~6개)"
          ],
          "questions_for_seller": [
            "판매자/딜러에게 공통으로 꼭 물어봐야 할 질문 (3~6개)"
          ],
          "recommendation": "최종적으로 어떻게 선택하는 게 좋을지에 대한 한두 문장 조언",

          "ranking": [
            {
              "index": 0,
              "short_title": "vehicles[0]에 해당하는 매물을 한 줄로 설명",
              "fit_score": 4.0,
              "fit_reason": "이 페르소나에게 왜 잘 맞는지 (2~4문장 정도)",
              "pros": ["이 페르소나 기준 장점 리스트"],
              "cons": ["이 페르소나 기준 단점/주의사항 리스트"]
            }
          ]
        }
        """).strip()

        persona_block = f"""
        [persona]
        id: {persona.id}
        label: {persona.label}
        description: {persona.description}
        """.strip()

        user_note_block = ""
        if has_user_note:
            user_note_block = f"""
            [사용자 메모]
            사용자가 직접 적은 걱정/조건입니다. ranking, fit_score, fit_reason에 반영하세요.

            \"\"\"{user_note.strip()}\"\"\" 
            """.strip()

        vehicles_block = f"""
        [vehicles]
        아래는 여러 매물에 대한 구조화된 정보 목록입니다. (파이썬 리스트/JSON 배열 형태)

        {vehicles_json}
        """.strip()

        blocks = [base_instruction, persona_block]
        if has_user_note:
            blocks.append(user_note_block)
        blocks.append(vehicles_block)
        return "\n\n".join(blocks)

        # ---------- B. 단일 매물용 ----------
    vehicle_json = json.dumps(vehicle_data, ensure_ascii=False, indent=2)

    if persona.mode == "buy":
        # 1) 예산과 무관한 공통 규칙
        base_instruction = textwrap.dedent("""
        당신은 중고차를 처음 보거나 익숙하지 않은 일반 사용자를 도와주는
        "중고차 구매 코치"입니다.

        아래 [persona]는 이 매물을 보는 사람의 상황/목적/성향을 설명합니다.
        아래 [vehicle]은 이 사람이 보고 있는 한 대의 매물에 대한 구조화된 정보입니다.

        원칙:
        - 항상 persona의 관점에서 설명하세요.
        - 자동차/보험/정비 전문 용어는 필요한 만큼만 쓰고, 짧게 풀어서 설명하세요.
        - vehicle_data에 없는 정보(보험료, 세금, 실제 연비, 정확한 유지비 등)는
          추측해서 단정하지 말고, "이 정보만으로는 정확히 알 수 없다"고 분명히 말하세요.
          다만 일반적인 경향은 "보통 ~인 경우가 많다" 수준으로만 언급하세요.

        출력 형식:
        반드시 아래 JSON 형식의 "하나의 객체"만 출력하세요.
        JSON 코드 블록이나 ```json 같은 래핑 없이, 순수 JSON만 출력하세요.

        {
          "mode": "buy",
          "persona_id": "...",
          "persona_label": "...",

          "summary": "...",
          ...
        }

        추가 규칙:
        - 불필요하게 장황하게 쓰지 말고, 핵심만 간결하게 정리하세요.
        - persona에 따라 정말 중요한 포인트 위주로 정리하세요.
        """).strip()

        # 2) 예산 유무에 따라 별도 블록 추가
        if has_budget:
            budget_block = textwrap.dedent("""
            예산 관련 규칙 (중요):
            - [vehicle]의 price_krw 필드에는 이 매물의 가격(원 단위)이 들어 있습니다.
            - [사용자 메모]에 적힌 예산 상한을 기준으로,
              price_krw가 이 예산을 넘는다면
              "예산보다 비싸다", "예산을 초과한다"라고 분명히 적으세요.
            - 예산을 넘더라도 다른 장점 때문에 추천할 수는 있지만,
              그 경우에도 "예산 상으로는 부담"이라는 표현을 반드시 포함하세요.
            """).strip()
        else:
            budget_block = textwrap.dedent("""
            예산 관련 규칙 (중요):
            - 이번 질문에서는 [사용자 메모]에 구체적인 예산 정보가 없습니다.
            - 사용자 예산을 임의로 추정하거나,
              "예산에 맞지 않는다", "예산을 초과한다" 같은 표현은 사용하지 마세요.
            - 대신 동급 평균 시세나 market_price_hint 를 활용하여
              "동급 시세 대비 비싸다/저렴하다" 수준으로만 가격을 평가하세요.
            """).strip()

        base_instruction = base_instruction + "\n\n" + budget_block


    else:
        base_instruction = textwrap.dedent("""
        당신은 중고차를 판매하려는 사람에게 조언해주는 "중고차 판매 코치"입니다.

        아래 [persona]는 판매자의 상황/목표/성향을 설명합니다.
        아래 [vehicle]은 판매하려는 차량 한 대에 대한 구조화된 정보입니다.

        출력 규칙(중요):
        - 반드시 하나의 JSON 객체만 출력하세요.
        - "요약", "장점" 같은 제목/설명 문장을 JSON 바깥에 쓰지 마세요.
        - JSON 코드 블록이나 ```json 같은 래핑 없이, 순수 JSON만 출력하세요.

        JSON 스키마는 아래와 같습니다. key 이름과 구조를 그대로 따르세요.

        {
          "mode": "sell",
          "persona_id": "...",
          "persona_label": "...",

          "summary": "이 차량을 어떻게 포지셔닝해서 팔면 좋을지 한 문단 정도로 요약",
          "fit_score": 0.0,
          "pros": ["판매 시 강조하면 좋을 점"],
          "cons": ["솔직하게 밝혀야 할 단점/주의사항"],
          "risk_level": "low | medium | high",
          "recommendation": "가격·채널·전략에 대한 한두 문장 조언",

          "listing_title": "중고차 사이트에 올릴 한 줄 제목 (최대 40자 이내, 과장/허위 없이 사실 위주)",
          "listing_body": "실제 중고차 사이트에 복붙해서 쓸 수 있는 소개 문구 3~6줄. 구매자가 읽는 글이므로 '빠른 판매', '현금화', '빨리 팔고 싶은 분' 같은 표현은 쓰지 말고, '빠르게 구매하고 싶으신 분께 추천드립니다', '편하게 구매를 진행하고 싶으신 분께 적합한 차량입니다'처럼 **구매자 입장**에서 자연스럽게 작성하세요."
        }

        추가 규칙:
        - listing_title, listing_body는 반드시 비워두지 말고 최소 한 문장 이상 채우세요.
        - listing_body 마지막 문장은 가능하면
          "빠르게 구매하고 싶으신 분께 추천드립니다." 또는
          "편하게 구매를 진행하고 싶으신 분께 잘 맞습니다."
          같은 형태로 **구매자 시점**으로 마무리하세요.
        """).strip()


    persona_block = f"""
    [persona]
    id: {persona.id}
    label: {persona.label}
    description: {persona.description}
    """.strip()

    user_note_block = ""
    if has_user_note:
        user_note_block = f"""
        [사용자 메모]
        아래 텍스트는 사용자가 직접 적은 메모입니다.
        이 사람이 무엇을 걱정하는지/중요하게 보는지를 파악하는 데 사용하세요.

        \"\"\"{user_note.strip()}\"\"\" 
        """.strip()

    vehicle_block = f"""
    [vehicle]
    아래는 한 대의 중고차 매물에 대한 구조화된 정보입니다. (JSON 객체 형태)

    {vehicle_json}
    """.strip()

    blocks = [base_instruction, persona_block]
    if has_user_note:
        blocks.append(user_note_block)
    blocks.append(vehicle_block)

    return "\n\n".join(blocks)


# ==============================
# 2-1. 여러 매물 비교 프롬프트 (메인 멀티용)
# ==============================

def build_multi_prompt(
    vehicle_list: List[Dict[str, Any]],
    persona: Persona,
    user_note: Optional[str] = None,
) -> str:
    """
    여러 매물을 한 번에 받아서 비교/랭킹하도록 하는 프롬프트.
    - Top1 매물만 상세(장점/단점/질문)
    - 나머지 매물은 index + title (+ fit_score 정도만)
    """
    has_user_note = bool(user_note and user_note.strip())
    has_budget = _has_budget(user_note)

    # 매물들을 [매물 1] ... [매물 N] 블록으로 펼쳐서 넣기 (멀티용 압축 포함)
    vehicles_block_parts = []
    for idx, v in enumerate(vehicle_list, start=1):
        v_short = _shrink_vehicle_for_multi(v)
        v_json = json.dumps(v_short, ensure_ascii=False, indent=2)
        vehicles_block_parts.append(f"[매물 {idx}]\n{v_json}")

    vehicles_block = "\n\n".join(vehicles_block_parts)

    # 공통 persona 블록
    persona_block = textwrap.dedent(f"""
    [persona]
    id: {persona.id}
    label: {persona.label}
    description: {persona.description}
    """).strip()

    # 사용자 메모 블록 (있을 때만)
    user_note_block = ""
    if has_user_note:
        user_note_block = textwrap.dedent(f"""
        [사용자 메모]
        아래 텍스트는 사용자가 직접 적은 메모입니다.
        이 사람이 무엇을 걱정하는지/중요하게 보는지를 파악하는 데 사용하세요.

        \"\"\"{user_note.strip()}\"\"\" 
        """).strip()

    if persona.mode == "buy":
        base_instruction = textwrap.dedent("""
        당신은 여러 중고차 매물 중에서,
        특정 사용자(persona)에게 가장 잘 맞는 매물을 골라주는 "중고차 구매 의사결정 코치"입니다.

        아래 persona 는 이 매물을 보는 사람의 상황/목적/성향을 설명합니다.
        아래 [매물 목록] 은 서로 다른 매물들의 요약 정보입니다.

        원칙:
        - 항상 persona 에 나와 있는 관점에서 생각하세요.
        - 자동차/보험/정비 전문 용어를 남발하지 말고, 필요하면 짧게 풀어서 설명하세요.
        - 각 매물의 장단점을 "persona에게 얼마나 맞는지" 관점에서 비교하세요.
        - 매물 정보에 없는 항목(보험료, 세금, 정확한 유지비 등)은 일반적인 경향만 말하고
          구체적인 숫자는 만들지 마세요.

        예산 관련 규칙 (중요):
        - 각 매물의 price_krw 필드에는 가격(원 단위)이 들어 있습니다.
        - [사용자 메모]에 예산 관련 사항이 있는 경우에만, price_krw가 이 예산을 넘는다면,
          summary나 highlights에서 "예산에 맞는다", "가격이 적당하다"라고 말하지 말고,
          "예산보다 비싸다", "예산을 초과한다"라고 분명히 적으세요.
        - 예산을 넘지만 다른 장점(연식, 주행거리, 사고 이력 등) 때문에 fit_score 가 높을 수는 있지만,
          그 경우에도 "예산 상으로는 부담"이라는 뉘앙스를 반드시 포함하세요.

        출력 형식 (JSON 하나만, 코드블록 금지):

        {
          "mode": "buy",
          "persona_id": "...",
          "persona_label": "...",

          "summary_overall": "여러 매물 비교 요약 (1~2문장, 80자 이내)",

          "best_index": 1,

          "best": {
            "index": 1,                    // [매물 목록]에서의 번호
            "title": "가장 잘 맞는 매물 제목",
            "fit_score": 0.0,              // 0.0 ~ 10.0
            "summary": "이 매물이 persona에게 어떤 느낌인지 1~2문장 (80자 이내)",
            "pros": ["장점 최대 3개"],
            "cons": ["단점/주의사항 최대 3개"],
            "questions_for_seller": ["판매자/딜러에게 물어볼 질문 최대 3개"],
            "risk_level": "low | medium | high"
          },

          "ranking": [
            {
              "index": 1,                  // [매물 목록] 번호
              "title": "매물 1의 제목",
              "fit_score": 0.0             // 상대적인 적합도 (0.0~10.0), 선택 사항이지만 가능하면 채우기
            }
          ]
        }

        규칙:
        - best 에 대해서만 pros/cons/questions_for_seller 를 작성합니다.
        - ranking 에서는 각 매물의 index, title, fit_score 만 작성합니다.
          (fit_score 가 애매하면 0.0~10.0 범위에서 대략적인 상대값만 줘도 됩니다.)
        - 전체 한국어 텍스트는 600자 이내로 쓰세요.
        - JSON 구조를 끝까지 완성하는 것이 가장 중요합니다.
          내용이 애매하면 빈 문자열("") 또는 짧은 문장으로 처리하세요.
        """).strip()
        
        if has_budget:
            budget_block = textwrap.dedent("""
            예산 하드 가드레일 (매우 중요):

            1) 예산 정보
            - [사용자 메모]에 적힌 예산 상한을 기준으로,
            각 매물의 price_krw 가 예산 이내(<=)인지 예산 초과(>)인지 먼저 판단하세요.

            2) 예산 이내 매물이 하나라도 있는 경우
            - best_index 는 반드시 예산 이내 매물들 중에서만 선택해야 합니다.
            - 예산 이내 매물들끼리 persona 관점에서 비교해서
            fit_score 를 0.0~10.0 사이로 주고,
            그중 가장 잘 맞는 한 대를 best 로 선택하세요.
            - 예산을 초과하는 매물은 ranking 에 포함해도 되지만,
            fit_score 는 최대 6.0까지만 주고,
            summary_overall 이나 best.summary 에서
            "최종 추천"처럼 보이게 쓰지 마세요.
            (예: "그래도 이 차가 더 낫다" 같은 표현 금지)

            3) 모든 매물이 예산을 초과하는 경우
            - summary_overall 첫 문장에
            "사용자가 적어주신 예산에 맞는 매물은 없고, 현재 매물은 모두 예산을 초과한다"
            는 내용을 반드시 포함하세요.
            - 각 매물에 대해 fit_score 를 0.0~10.0 범위에서 모두 부여하고,
            그중 상대적으로 조건이 나은 매물을 best_index 로 선택하세요.
            - 이때 best.summary 에도
            "예산을 초과하지만" 또는 비슷한 표현을 꼭 넣으세요.

            4) 표현 규칙
            - 예산을 초과하는 매물에 대해서는
            summary_overall, best.summary, pros 어디에서도
            "예산에 잘 맞는다", "가격이 부담되지 않는다" 같은 표현을 쓰지 마세요.
            """).strip()
        else:
            budget_block = textwrap.dedent("""
            예산 관련 규칙 (중요):
            - 이번 질문에서는 [사용자 메모]에 구체적인 예산 정보가 없습니다.
            - 사용자 예산을 추정하거나,
            "예산에 맞는다", "예산에 맞지 않는다", "예산을 초과한다" 같은 표현은 사용하지 마세요.
            - 가격 언급은 "동급 시세 대비 비싸다/저렴하다"와 같이
            상대적인 시세 기준으로만 설명하세요.
            """).strip()

        base_instruction = base_instruction + "\n\n" + budget_block


    else:
        base_instruction = textwrap.dedent("""
        당신은 여러 대의 차량을 가진 판매자가
        어떤 차량을 먼저 팔거나 어떻게 전략을 잡으면 좋을지 도와주는
        "중고차 판매 전략 코치"입니다.

        아래 persona 는 판매자의 상황/목표/성향을 설명합니다.
        아래 [매물 목록] 은 판매자가 보유한 서로 다른 차량 정보입니다.

        출력 형식 (JSON 하나만, 코드블록 금지):

        {
          "mode": "sell",
          "persona_id": "...",
          "persona_label": "...",

          "summary_overall": "여러 차량을 어떤 순서/전략으로 판매하면 좋을지 한두 문장 요약",

          "best_index": 1,

          "best": {
            "index": 1,
            "title": "먼저 팔면 좋은 차량 제목",
            "fit_score": 0.0,
            "summary": "왜 이 차량을 먼저 파는 게 좋은지 1~2문장",
            "pros": ["판매 시 강조하면 좋을 점 (최대 3개)"],
            "cons": ["솔직하게 밝혀야 할 단점/주의사항 (최대 3개)"],
            "questions_for_seller": ["거래 과정에서 특히 확인해야 할 사항 (최대 3개)"],
            "risk_level": "low | medium | high"
          },

          "ranking": [
            {
              "index": 1,
              "title": "차량 제목",
              "fit_score": 0.0
            }
          ]
        }

        규칙:
        - best 에 대해서만 pros/cons/questions_for_seller 를 작성합니다.
        - ranking 은 index, title, fit_score 정도만 간단히 적으세요.
        - 전체 한국어 텍스트는 600자 이내로 쓰세요.
        """).strip()

    if has_user_note:
        extra = textwrap.dedent("""
        [중요]

        아래 [사용자 메모]에 사용자가 직접 적은 걱정/조건이 있다면,
        summary_overall, best.summary/pros/cons/questions_for_seller,
        ranking[*].fit_score 에 자연스럽게 반영하세요.

        단, 매물 정보에 없는 속성에 대해서는
        - '정보가 없어서 정확히 비교는 어렵다'고 언급하거나,
        - 일반적인 경향 수준으로만 조심스럽게 설명하세요.
        """).strip()
        instruction = base_instruction + "\n\n" + extra
    else:
        instruction = base_instruction

    blocks = [instruction, persona_block]
    if has_user_note:
        blocks.append(user_note_block)
    blocks.append("[매물 목록]\n" + vehicles_block)

    return "\n\n".join(blocks)




# ==============================
# 3. LLM 로딩 & 호출 (Mi:dm 2.0)
# ==============================

MODEL_ID_DEFAULT = os.getenv("MIDM_MODEL", "K-intelligence/Midm-2.0-Base-Instruct")

_tokenizer = None
_model = None
_loaded_model_id = None


def _load_model(model_id: str = MODEL_ID_DEFAULT):
    """
    Mi:dm 2.0 모델 lazy-load.
    - GPU 가 있으면 float16 + device_map="auto"
    - 없으면 CPU float32 로 로드 (MIDM_FORCE_CPU=1 도 강제 CPU)
    """
    global _tokenizer, _model, _loaded_model_id

    if _model is not None and _loaded_model_id == model_id:
        return

    print(f"[Mi:dm] loading model: {model_id}")

    force_cpu = os.getenv("MIDM_FORCE_CPU", "0") == "1"
    has_cuda = torch.cuda.is_available() and not force_cpu

    if has_cuda:
        torch_dtype = torch.float16
        device_map = "auto"
        print("[Mi:dm] using GPU (float16, device_map=auto)")
    else:
        torch_dtype = torch.float32
        device_map = None
        print("[Mi:dm] using CPU (float32)")

    _tokenizer = AutoTokenizer.from_pretrained(model_id)

    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    ).eval()

    print("[Mi:dm] device:", _model.device)
    _loaded_model_id = model_id


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """
    Mi:dm 2.0 호출 래퍼.
    - system 역할에 "JSON만 출력" 규칙을 강하게 명시
    - chat_template + add_generation_prompt=True 사용
    """
    global _tokenizer, _model

    model_id = model or MODEL_ID_DEFAULT
    _load_model(model_id)

    system_prompt = (
        "너는 중고차 매물 정보를 분석해서 JSON 형식으로만 응답하는 엔카 코파일럿이다. "
        "반드시 하나의 JSON 객체만 출력해야 하며, '요약', '장점' 같은 제목이나 다른 설명 문장은 "
        "JSON 바깥에 절대 출력하지 마라. JSON 코드 블록이나 ```json 같은 래핑도 사용하지 마라."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    input_ids = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,              # JSON 뽑을 거라 sampling 끔
            temperature=0.0,      # 혹시라도 사용할 경우 대비
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.eos_token_id,
            top_p = 1.0
        )

    gen_ids = outputs[0][input_ids.shape[1]:]
    print(f"[DEBUG] generated tokens: {gen_ids.shape[0]} (max_new_tokens={max_new_tokens})")

    text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


# ==============================
# 4. LLM 결과 JSON 파싱 유틸
# ==============================

def _strip_code_fence(txt: str) -> str:
    """ ```json ... ``` 같은 래핑 제거 """
    txt = txt.strip()
    if txt.startswith("```"):
        parts = txt.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            inner = inner.lstrip()
            if inner.lower().startswith("json"):
                inner = inner[4:]
            return inner.strip()
    return txt


def _strip_reasoning_wrappers(txt: str) -> str:
    """
    DeepSeek / Mi:dm 류 모델이 <think>...</think> 이나
    ```json ...``` 으로 감싸서 줄 때 제거.
    """
    import re
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    txt = re.sub(r"(?s)```(?:json)?(.*?)```", r"\1", txt)
    return txt.strip()


def _safe_json_extract(txt: str) -> Dict[str, Any]:
    """
    LLM이 출력한 텍스트에서 JSON dict를 최대한 안전하게 뽑아낸다.
    - 우리가 기대하는 '결과 JSON'처럼 생긴 dict만 채택한다.
    - 그런 게 하나도 없으면 {"raw_text": ...} 로 fallback 한다.
    """
    import re
    import json as _json

    txt = _strip_reasoning_wrappers(txt or "")
    txt = _strip_code_fence(txt)

    # "이런 키들 중 하나라도 포함되면 '결과 JSON'이라고 본다"
    EXPECTED_KEYS = (
        "summary",
        "summary_overall",
        "ranked_candidates",
        "fit_score",
        "pros",
        "cons",
        "best_index",
    )

    def looks_like_result(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        return any(k in obj for k in EXPECTED_KEYS)

    # 1차: 전체 문자열 그대로 시도
    try:
        obj = _json.loads(txt)
        if looks_like_result(obj):
            return obj
    except Exception:
        pass

    # 2차: { ... } 블록만 추출
    candidates: List[str] = []
    depth = 0
    start = None

    for i, ch in enumerate(txt):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(txt[start: i + 1])

    # 아예 { } 가 하나도 없는 경우: 완전 비JSON → fallback
    if not candidates:
        return {"raw_text": txt.strip()}

    # 3차: 뒤에서부터(마지막 JSON부터) 파싱
    for cand in reversed(candidates):
        body_clean = cand.strip()

        # 3-1) trailing comma 제거: ... ,] / ,} 형태
        body_clean = re.sub(r",(\s*[\]\}])", r"\1", body_clean)

        # 3-2) 전형적인 파이썬 dict 스타일 '{...}' 인 경우, ' 를 " 로 치환
        if '"' not in body_clean and "'" in body_clean:
            body_clean = body_clean.replace("'", '"')

        try:
            obj = _json.loads(body_clean)
            if looks_like_result(obj):
                return obj
        except Exception:
            continue

    # 여기까지 왔다는 건, JSON은 있긴 했는데
    # 우리가 원하는 형태(summary, ranked_candidates 등)는 아니었다는 뜻.
    # → 그냥 raw 텍스트 통째로 넘기자.
    return {"raw_text": txt.strip()}



# ==============================
# 5. 결과 정규화 도우미
# ==============================

def _normalize_risk_level(value: Any) -> str:
    if not isinstance(value, str):
        return "medium"
    v = value.strip().lower()
    if v in ("low", "중간이하", "낮음"):
        return "low"
    if v in ("high", "높음"):
        return "high"
    return "medium"


def _clamp_float(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    if f < lo:
        return lo
    if f > hi:
        return hi
    return f


def _normalize_single_result(
    parsed: Dict[str, Any],
    mode: Mode,
    persona: Persona,
) -> Dict[str, Any]:
    """
    단일 매물 결과를 최소한의 스키마로 정리.
    """
    parsed = dict(parsed)  # shallow copy

    parsed.setdefault("mode", mode)
    parsed.setdefault("persona_id", persona.id)
    parsed.setdefault("persona_label", persona.label)

    parsed.setdefault("summary", "")
    parsed["fit_score"] = _clamp_float(parsed.get("fit_score", 0.0), 0.0, 10.0, 0.0)
    parsed["risk_level"] = _normalize_risk_level(parsed.get("risk_level", "medium"))

    # 공통 필드 기본값
    if not isinstance(parsed.get("highlights"), list):
        parsed["highlights"] = []
    if not isinstance(parsed.get("pros"), list):
        parsed["pros"] = []
    if not isinstance(parsed.get("cons"), list):
        parsed["cons"] = []
    if not isinstance(parsed.get("checklist"), list):
        parsed["checklist"] = []
    if not isinstance(parsed.get("questions_for_seller"), list):
        parsed["questions_for_seller"] = []
    if not isinstance(parsed.get("recommendation"), str):
        parsed["recommendation"] = ""

    # 🔹 판매 모드일 때 listing_* 기본값 확보
    if mode == "sell":
        if not isinstance(parsed.get("listing_title"), str):
            parsed["listing_title"] = ""
        if not isinstance(parsed.get("listing_body"), str):
            parsed["listing_body"] = ""

    return parsed

def _normalize_multi_result(
    parsed: Dict[str, Any],
    vehicle_count: int,
    mode: Mode,
    persona: Persona,
) -> Dict[str, Any]:
    parsed = dict(parsed)

    parsed.setdefault("mode", mode)
    parsed.setdefault("persona_id", persona.id)
    parsed.setdefault("persona_label", persona.label)
    parsed.setdefault("summary_overall", "")
    parsed.setdefault("tradeoffs", [])

    if not isinstance(parsed.get("tradeoffs"), list):
        parsed["tradeoffs"] = [str(parsed.get("tradeoffs", ""))]

    # 🔥 1) LLM이 best + ranking 형태로 줄 때 보정
    if "best" in parsed and (
        "ranked_candidates" not in parsed
        or not isinstance(parsed.get("ranked_candidates"), list)
        or not parsed.get("ranked_candidates")
    ):
        best = parsed.get("best")
        ranking = parsed.get("ranking", [])

        ranked_candidates: List[Dict[str, Any]] = []
        best_index_val = None

        if isinstance(best, dict):
            try:
                best_index_val = int(best.get("index", 1))
            except Exception:
                best_index_val = 1

        if isinstance(ranking, list) and ranking:
            for item in ranking:
                if not isinstance(item, dict):
                    continue
                c = dict(item)
                idx_val = None
                try:
                    idx_val = int(c.get("index", 0))
                except Exception:
                    pass

                # best와 index가 같으면 정보 merge
                if isinstance(best, dict) and best_index_val is not None and idx_val == best_index_val:
                    for key in (
                        "summary",
                        "pros",
                        "cons",
                        "checklist",
                        "questions_for_seller",
                        "risk_level",
                        "why_suitable",
                    ):
                        if key in best and key not in c:
                            c[key] = best[key]
                ranked_candidates.append(c)
        elif isinstance(best, dict):
            # ranking이 없고 best만 있는 경우
            ranked_candidates = [dict(best)]

        if ranked_candidates:
            parsed["ranked_candidates"] = ranked_candidates
            if "best_index" not in parsed and best_index_val is not None:
                parsed["best_index"] = best_index_val

        # 전체 risk_level이 비어있으면 best 기준으로 올려주기 (캡션용)
        if "risk_level" not in parsed and isinstance(best, dict) and best.get("risk_level"):
            parsed["risk_level"] = best["risk_level"]

    # 🔥 2) 여기부터는 기존 로직 (후보 정규화)
    cands = parsed.get("ranked_candidates", [])
    if not isinstance(cands, list):
        cands = []

    norm_cands = []
    for idx, c in enumerate(cands):
        if not isinstance(c, dict):
            continue
        c = dict(c)

        index = c.get("index", idx + 1)
        try:
            index_int = int(index)
        except Exception:
            index_int = idx + 1
        if index_int < 1 or index_int > vehicle_count:
            index_int = max(1, min(vehicle_count, index_int))
        c["index"] = index_int

        c.setdefault("title", "")
        c.setdefault("summary", "")
        if not isinstance(c.get("pros"), list):
            c["pros"] = []
        if not isinstance(c.get("cons"), list):
            c["cons"] = []
        if not isinstance(c.get("checklist"), list):
            c["checklist"] = []
        if not isinstance(c.get("questions_for_seller"), list):
            c["questions_for_seller"] = []

        # ✅ checklist 없으면 questions_for_seller로라도 채워넣기
        if not c["checklist"] and c["questions_for_seller"]:
            c["checklist"] = c["questions_for_seller"][:3]

        c["fit_score"] = _clamp_float(c.get("fit_score", 0.0), 0.0, 10.0, 0.0)
        c["risk_level"] = _normalize_risk_level(c.get("risk_level", "medium"))
        c.setdefault("why_suitable", "")

        norm_cands.append(c)

    norm_cands.sort(key=lambda x: x.get("fit_score", 0.0), reverse=True)
    parsed["ranked_candidates"] = norm_cands

    best_index = parsed.get("best_index", None)
    try:
        best_index_int = int(best_index)
    except Exception:
        best_index_int = 1

    if best_index_int < 1 or best_index_int > vehicle_count:
        best_index_int = 1
    parsed["best_index"] = best_index_int

    return parsed





# ==============================
# 6. 외부에서 호출할 메인 함수
# ==============================

def generate_view(
    vehicle_data: Dict[str, Any],
    persona_id: str,
    mode: Mode = "buy",
    model: Optional[str] = None,
    persona_obj: Optional[Persona] = None,
    user_note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    단일 매물용 진입점.
    - vehicle_data: 단일 매물 dict
    - persona_id + mode 로 Persona 선택 (또는 persona_obj 직접 전달)
    """
    if persona_obj is not None:
        persona = persona_obj
    else:
        persona = get_persona(persona_id, mode)

    prompt = build_prompt(vehicle_data, persona, user_note=user_note)
    raw = call_llm(prompt, model=model, max_new_tokens = 512)

    print("[generate_view] RAW LLM OUTPUT:")
    print(raw)

    parsed = _safe_json_extract(raw)
    parsed = _normalize_single_result(parsed, mode, persona)
    return parsed


def generate_multi_view(
    vehicle_list: List[Dict[str, Any]],
    persona_id: str,
    mode: Mode = "buy",
    model: Optional[str] = None,
    persona_obj: Optional[Persona] = None,
    user_note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    여러 매물에 대해 비교/랭킹을 수행하는 진입점 함수.
    """
    if not vehicle_list:
        raise ValueError("vehicle_list 가 비어 있습니다.")

    if persona_obj is not None:
        persona = persona_obj
    else:
        persona = get_persona(persona_id, mode)

    prompt = build_multi_prompt(vehicle_list, persona, user_note=user_note)
    raw = call_llm(
        prompt,
        model=model,
        max_new_tokens=512,   # ✅ 512면 충분하도록 프롬프트를 줄여놨음
        temperature=0.0,
    )

    print("[generate_multi_view] RAW LLM OUTPUT:")
    print(raw)

    parsed = _safe_json_extract(raw)
    parsed = _normalize_multi_result(
        parsed,
        vehicle_count=len(vehicle_list),
        mode=mode,
        persona=persona,
    )
    return parsed



# ==============================
# 7. 간단 CLI 테스트용
# ==============================

if __name__ == "__main__":
    sample_vehicle1 = {
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
            "후측방경보",
        ],
        "inspection": {
            "encar_inspection": "엔카진단+",
            "comments": "외관 경미한 스톤칩, 하부 부식 없음, 타이어 마모 40% 정도 남음",
        },
        "market_price_hint": "동급 평균 시세 대비 약간 낮은 편",
    }

    sample_vehicle2 = {
        "title": "K5 DL3 2.0 가솔린 노블레스",
        "year": 2020,
        "mileage_km": 62000,
        "price_krw": 10900000,
        "color": "핑크색",
        "accident_history": "무사고, 단순판금 도색 있음",
        "usage_history": "개인 출퇴근용 4년",
        "options": [
            "크루즈컨트롤",
            "차선이탈경고",
            "열선시트",
            "전방주차센서",
        ],
        "inspection": {
            "encar_inspection": "엔카진단",
            "comments": "외관 스크래치 일부, 하부 부식 없음, 타이어 마모 30% 정도 남음",
        },
        "market_price_hint": "동급 평균 시세와 비슷한 편",
    }

    persona = get_persona("first_car_student", "buy")

    print("=== SAMPLE PROMPT (single) ===")
    print(build_prompt(sample_vehicle1, persona))

    print("\n=== SAMPLE PROMPT (multi) ===")
    print(build_multi_prompt([sample_vehicle1, sample_vehicle2], persona))

    # 실제 LLM 호출 테스트 (모델 설치/환경 필요)
    # result_single = generate_view(sample_vehicle1, "first_car_student", "buy")
    # print("\n=== RESULT SINGLE ===")
    # print(json.dumps(result_single, ensure_ascii=False, indent=2))

    # result_multi = generate_multi_view([sample_vehicle1, sample_vehicle2], "first_car_student", "buy")
    # print("\n=== RESULT MULTI ===")
    # print(json.dumps(result_multi, ensure_ascii=False, indent=2))
