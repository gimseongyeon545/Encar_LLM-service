# streamlit_app.py
import json
from typing import Dict, Any, Optional, List

import streamlit as st
import re

from inference import (generate_view,
    generate_multi_view,
    BUY_PERSONAS,
    SELL_PERSONAS,
    Persona,
    )


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="🚗엔카 역할 기반 코파일럿 (프로토타입)",
    layout="wide",
)

st.title("🚗엔카 역할 기반 코파일럿 (프로토타입)")
st.markdown(
    """
🚗엔카 차량 상세 정보와 **사용자 역할(페르소나)**를 입력으로 받아,  
역할에 맞는 요약 / 장단점 / 체크리스트를 생성하는 LLM 기반 도우미입니다.

- 기본: 한 대의 매물에 대해 페르소나 맞춤 분석  
- 고급: 여러 매물을 JSON 배열로 입력하면, 페르소나 관점에서 비교·순위 추천 (차 사기 모드 기준)
"""
)


# =========================
# 0. 샘플 차량 데이터
# =========================
DEFAULT_VEHICLE: Dict[str, Any] = {
    "title": "쏘나타 DN8 2.0 가솔린 프리미엄",
    "year": 2021,
    "mileage_km": 48000,
    "price_krw": 18500000,
    "color": "화이트",
    "accident_history": "앞펜더 단순교환 1회, 프레임 손상 없음",
    "usage_history": "렌트 이력 1년, 이후 개인 자가용 2년",
    "options": ["스마트크루즈", "차선이탈보조", "통풍시트", "후측방경보"],
    "inspection": {
        "encar_inspection": "엔카진단+",
        "comments": "외관 경미한 스톤칩, 하부 부식 없음, 타이어 마모 40% 정도 남음",
    },
    "market_price_hint": "동급 평균 시세 대비 약간 낮은 편",
}


# 세션 상태 초기화
if "vehicle_data" not in st.session_state:
    st.session_state["vehicle_data"] = DEFAULT_VEHICLE
    st.session_state["vehicle_confirmed"] = False

if "vehicle_json_text" not in st.session_state:
    st.session_state["vehicle_json_text"] = json.dumps(
        st.session_state["vehicle_data"],
        ensure_ascii=False,
        indent=2,
    )

if "vehicle_list" not in st.session_state:
    st.session_state["vehicle_list"] = [DEFAULT_VEHICLE]


if "context_confirmed" not in st.session_state:
    st.session_state["context_confirmed"] = False
    st.session_state["saved_mode"] = "buy"
    st.session_state["saved_persona_id"] = None
    st.session_state["saved_custom_persona"] = None
    st.session_state["saved_user_note"] = ""

if "custom_persona_desc" not in st.session_state:
    st.session_state["custom_persona_desc"] = ""


# =========================
# 색상 유틸
# =========================
def _color_name_to_hex(color_name: str) -> str:
    """차량 색상 문자열을 대략적인 HEX 색상으로 매핑 (한국어/영어 몇 개만)"""
    if not color_name:
        return ""

    name = str(color_name).lower().strip()

    # 이미 hex나 rgb로 들어온 경우 그대로 사용
    if name.startswith("#") and len(name) in (4, 7):
        return name
    if name.startswith("rgb"):
        return name

    # 화이트 계열
    if "white" in name or "화이트" in name:
        return "#e5e7eb"
    # 블랙/검정
    if "black" in name or "블랙" in name or "검정" in name:
        return "#111827"
    # 그레이/실버
    if ("silver" in name or "실버" in name or
        "grey" in name or "gray" in name or
        "그레이" in name or "회색" in name):
        return "#9ca3af"
    # 블루
    if "blue" in name or "블루" in name or "파랑" in name or "파란" in name:
        return "#2563eb"
    # 레드/와인
    if ("red" in name or "레드" in name or "빨강" in name or
        "와인" in name or "버건디" in name):
        return "#dc2626"
    # 주황/오렌지
    if "orange" in name or "오렌지" in name or "주황" in name:
        return "#f97316"
    # 핑크/분홍/로즈
    if ("핑크" in name or "분홍" in name or
        "pink" in name or "로즈" in name):
        return "#ec4899"
    # 그린
    if "green" in name or "그린" in name or "초록" in name:
        return "#16a34a"
    # 베이지/골드
    if "beige" in name or "베이지" in name or "골드" in name or "gold" in name:
        return "#d6d3d1"

    # 그 외는 그냥 연한 회색
    return "#d1d5db"



# =========================
# 차량 카드 UI
# =========================
def render_vehicle_card(data: Dict[str, Any]):
    """엔카 스타일 가벼운 카드 UI (색상 뱃지 포함)"""
    title = data.get("title", "차량 제목 미입력")
    year = data.get("year", "-")
    mileage = data.get("mileage_km", "-")
    color = data.get("color", "-")
    price = data.get("price_krw")
    accident = data.get("accident_history", "-")
    usage = data.get("usage_history", "-")
    market_hint = data.get("market_price_hint", "-")

    try:
        price_str = f"{int(price):,}원" if price is not None else "-"
    except Exception:
        price_str = str(price) if price is not None else "-"

    color_hex = _color_name_to_hex(color)
    color_dot = ""
    if color_hex:
        color_dot = f"""
        <span style="
            display:inline-block;
            width:10px;
            height:10px;
            border-radius:9999px;
            background:{color_hex};
            border:1px solid #9ca3af;
            margin-right:4px;
            vertical-align:middle;
        "></span>
        """

    html = f"""
    <div style="
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid #e5e7eb;
        background-color: #fefce8;
        margin-bottom: 10px;
    ">
      <div style="font-weight: 600; font-size: 1.05rem; margin-bottom: 4px; color:#111827;">
        {title}
      </div>
      <div style="font-size: 0.9rem; color: #4b5563;">
        <b>연식</b>: {year}년 &nbsp;|&nbsp;
        <b>주행거리</b>: {mileage} km
      </div>
      <div style="font-size: 0.95rem; margin-top: 4px; color:#111827;">
        <b>가격</b>: <span style="color: #b45309; font-weight: 700;">{price_str}</span>
      </div>
      <div style="font-size: 0.9rem; color: #4b5563; margin-top:4px;">
        <b>색상</b>: {color_dot}{color}
      </div>
      <div style="font-size: 0.85rem; color: #6b7280; margin-top: 8px;">
        <b>사고/이력</b>: {accident}<br/>
        <b>사용 이력</b>: {usage}<br/>
        <b>시세 힌트</b>: {market_hint}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================
# 사용자 상황 요약 카드
# =========================
import textwrap  # 파일 맨 위 import 쪽에 이 줄 추가

def render_user_context_card(
    mode: str,
    persona_label: Optional[str],
    persona_desc: Optional[str],
    user_note: Optional[str],
):
    """사용자 상황 요약 카드 UI (HTML 한 번에 렌더링)"""
    mode_label = "차 사기 (구매)" if mode == "buy" else "차 팔기 (판매)"

    persona_label = persona_label or ""
    persona_desc = persona_desc or ""
    user_note = (user_note or "").strip()

    st.markdown("### 사용자 요약 카드")

    html = f"""
<div style="border-radius: 12px; padding: 14px 16px; border: 1px solid #e5e7eb;
background-color: #fefce8; margin-bottom: 10px;">


  <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 6px; color:#111827;">
    사용자 상황 요약
  </div>
  <div style="font-size: 0.9rem; color: #4b5563; margin-bottom:4px;">
    <b>모드</b>: {mode_label}
  </div>
"""

    if persona_label:
        html += f"""
  <div style="font-size: 0.9rem; color: #4b5563; margin-bottom:4px;">
    <b>페르소나</b>: {persona_label}
  </div>
"""

    if persona_desc:
        html += f"""
  <div style="font-size: 0.82rem; color: #6b7280; margin-bottom:6px;">
    {persona_desc}
  </div>
"""

    if user_note:
        html += f"""
  <div style="font-size: 0.85rem; color: #374151; margin-top:4px;">
    <b>사용자 메모</b><br/>
    {user_note}
  </div>
"""

    html += "</div>"

    # 들여쓰기 제거해서 코드블럭으로 인식 안 되게
    html = textwrap.dedent(html)
    st.markdown(html, unsafe_allow_html=True)



# =========================
# 1. 차량 정보 입력 + 차량 카드
# =========================
vehicle_error = None

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 1. 차량 정보 입력")
    vehicle_json_text = st.text_area(
        "vehicle_data (엔카 상세에서 파싱한 JSON이라고 가정)",
        json.dumps(st.session_state["vehicle_data"], ensure_ascii=False, indent=2),
        height=260,
    )

    if st.button("1단계: 차량 정보 확인", key="confirm_vehicle"):
        try:
            parsed = json.loads(vehicle_json_text)

            # ✅ dict 한 개든, list 여러 개든 다 지원
            if isinstance(parsed, dict):
                vehicle_list = [parsed]
            elif isinstance(parsed, list):
                if not all(isinstance(v, dict) for v in parsed):
                    raise ValueError("리스트 안에는 차량 dict만 들어가야 합니다.")
                vehicle_list = parsed
            else:
                raise ValueError("vehicle_data는 dict 또는 dict 리스트여야 합니다.")

            st.session_state["vehicle_list"] = vehicle_list
            st.session_state["vehicle_data"] = vehicle_list[0]  # 대표(첫 번째) 매물
            st.session_state["vehicle_confirmed"] = True

            st.success(f"차량 정보 {len(vehicle_list)}개가 확인되었습니다.")
        except Exception as e:
            st.session_state["vehicle_confirmed"] = False
            vehicle_error = str(e)
            st.error(f"vehicle_data 파싱 오류: {e}")




with col_right:
    st.markdown("#### 차량 요약 카드")

    if st.session_state["vehicle_confirmed"]:
        vehicle_list = st.session_state.get("vehicle_list")

        # 혹시 vehicle_list가 없다면 예전 방식 fallback
        if not vehicle_list:
            render_vehicle_card(st.session_state["vehicle_data"])
        else:
            if len(vehicle_list) == 1:
                # 매물 1대면 그냥 한 개만
                render_vehicle_card(vehicle_list[0])
            else:
                st.caption(f"총 {len(vehicle_list)}대 매물")
                for idx, v in enumerate(vehicle_list, start=1):
                    title = v.get("title", f"매물 {idx}")
                    st.markdown(f"##### 매물 {idx}: {title}")
                    render_vehicle_card(v)
    else:
        st.info("왼쪽에서 차량 정보를 입력하고 '1단계: 차량 정보 확인' 버튼을 눌러주세요.")

st.markdown("---")

# =========================
# 2. 추가 정보 + 모드/페르소나 + 사용자 상황 카드
# =========================
col_info, col_summary = st.columns([2, 1])

with col_info:
    st.markdown("### 2. 추가 정보 (선택)")

    # 2-1) 모드 선택
    mode_label = st.radio(
        "모드 선택",
        ["차 사기 (구매)", "차 팔기 (판매)"],
        horizontal=True,
    )
    mode = "buy" if "사기" in mode_label else "sell"

    # 2-2) 페르소나 선택
    persona_table = BUY_PERSONAS if mode == "buy" else SELL_PERSONAS
    persona_id_to_label = {pid: p.label for pid, p in persona_table.items()}
    label_to_persona_id = {v: k for k, v in persona_id_to_label.items()}

    persona_labels = list(persona_id_to_label.values()) + ["기타 (직접 작성)"]

    persona_label_choice = st.selectbox(
        "페르소나 선택",
        persona_labels,
    )

    custom_persona_obj: Optional[Persona] = None

    if persona_label_choice == "기타 (직접 작성)":
        custom_desc_input = st.text_area(
            "나의 상태를 적어주세요.",
            st.session_state["custom_persona_desc"],
            height=120,
        )

        if st.button("페르소나 내용 확인", key="confirm_custom_persona"):
            st.session_state["custom_persona_desc"] = custom_desc_input
            st.success("페르소나 설명이 저장되었습니다.")

        final_desc = st.session_state["custom_persona_desc"].strip()
        persona_id = "custom"
        custom_label = final_desc or "사용자 정의 페르소나"

        custom_persona_obj = Persona(
            id="custom",
            label=custom_label,
            mode=mode,
            description=final_desc or "사용자가 직접 작성한 페르소나입니다.",
        )
    else:
        persona_id = label_to_persona_id[persona_label_choice]

    # 2-3) 추가 메모 (user_note)
    user_note = st.text_area(
        "추가로 걱정되거나 중요하게 보고 싶은 점이 있으면 적어주세요 (선택)",
        #placeholder="예: 첫 차라 보험료랑 주차가 특히 걱정돼요. 장거리 운전은 거의 안 합니다.",
        height=100,
    )

    # 2-4) 컨텍스트 확정 버튼
    if st.button("2단계: 모드/페르소나/추가정보 확인", key="confirm_context"):
        st.session_state["saved_mode"] = mode
        st.session_state["saved_persona_id"] = persona_id
        st.session_state["saved_custom_persona"] = custom_persona_obj
        st.session_state["saved_user_note"] = user_note
        st.session_state["context_confirmed"] = True
        st.success("모드/페르소나/추가정보가 반영되었습니다.")

with col_summary:
    if st.session_state["context_confirmed"]:
        saved_mode = st.session_state["saved_mode"]
        saved_persona_id = st.session_state["saved_persona_id"]
        saved_custom = st.session_state["saved_custom_persona"]
        saved_user_note = st.session_state["saved_user_note"]

        if saved_custom is not None:
            persona_label = saved_custom.label
            persona_desc = saved_custom.description
        else:
            p_table = BUY_PERSONAS if saved_mode == "buy" else SELL_PERSONAS
            if saved_persona_id is not None and saved_persona_id in p_table:
                p = p_table[saved_persona_id]
                persona_label = p.label
                persona_desc = p.description
            else:
                persona_label = None
                persona_desc = None

        render_user_context_card(
            mode=saved_mode,
            persona_label=persona_label,
            persona_desc=persona_desc,
            user_note=saved_user_note,
        )
    else:
        st.info("왼쪽에서 모드/페르소나/추가정보를 입력하고 '2단계' 버튼을 눌러주세요.")

st.markdown("---")



# =========================
# 3. LLM 호출 버튼 & 결과 표시
# =========================

run_disabled = (
    vehicle_error is not None
    or not st.session_state["vehicle_confirmed"]
    or not st.session_state["context_confirmed"]
)

if st.button("LLM 분석 실행", type="primary", disabled=run_disabled):
    if not st.session_state["vehicle_confirmed"]:
        st.error("먼저 1단계에서 차량 정보를 확인해 주세요.")
        st.stop()
    if not st.session_state["context_confirmed"]:
        st.error("먼저 2단계에서 모드/페르소나/추가정보를 확인해 주세요.")
        st.stop()

    # ✅ vehicle_list 기준으로 단일 vs 멀티 판단
    vehicle_list: List[Dict[str, Any]] = st.session_state.get("vehicle_list", [])
    if not vehicle_list:
        st.error("vehicle_list 가 비어 있습니다. 1단계에서 차량 정보를 다시 확인해 주세요.")
        st.stop()

    saved_mode = st.session_state["saved_mode"]
    saved_persona_id = st.session_state["saved_persona_id"]
    saved_custom = st.session_state["saved_custom_persona"]
    saved_user_note = st.session_state["saved_user_note"]

    # 수정: "사기(buy) + 2대 이상"일 때만 멀티 비교
    is_multi = (len(vehicle_list) > 1) and (saved_mode == "buy")


    with st.spinner("LLM 호출 중..."):
        try:
            if is_multi:
                # 여러 매물 비교
                result = generate_multi_view(
                    vehicle_list,
                    persona_id=saved_persona_id,
                    mode=saved_mode,
                    model=None,
                    persona_obj=saved_custom,
                    user_note=saved_user_note,
                )
            else:
                # 단일 매물
                result = generate_view(
                    vehicle_list[0],
                    persona_id=saved_persona_id,
                    mode=saved_mode,
                    model=None,
                    persona_obj=saved_custom,
                    user_note=saved_user_note,
                )
        except Exception as e:
            st.error(f"LLM 호출 또는 JSON 파싱 중 오류 발생: {e}")
            st.stop()

    st.markdown("### 3. LLM 결과")

    # 모델이 JSON을 안 지키고 raw_text만 넘어온 경우 대비
    raw_text = result.get("raw_text")
    if raw_text:
        with st.expander("⚠ 모델이 JSON 형식을 완전히 지키지 않았습니다. 원문 보기"):
            st.write(raw_text)

    # =========================
    # 💸 예산 파싱 & 체크 (buy 모드 전용)
    # =========================
    budget_max = None
    budget_warning_text = None

    if saved_mode == "buy":
        note = (saved_user_note or "").replace(",", "")
        matches = re.findall(r"(\d+)\s*(?:만|만원)\s*원?", note)
        if matches:
            try:
                # 여러 숫자 있으면 가장 작은 값을 '예산 상한'으로 봄
                max_unit = min(int(x) for x in matches)
                budget_max = max_unit * 10_000
            except Exception:
                budget_max = None

    if saved_mode == "buy" and budget_max is not None:
        # 추천 매물 가격 가져오기
        price_int = None
        try:
            if is_multi:
                best = result.get("best") or {}
                idx = best.get("index", result.get("best_index", 1))
                try:
                    idx_int = int(idx)
                except Exception:
                    idx_int = 1
                if not (1 <= idx_int <= len(vehicle_list)):
                    idx_int = 1
                best_vehicle = vehicle_list[idx_int - 1]
                price_int = int(best_vehicle.get("price_krw"))  # 실패하면 except 쪽으로
            else:
                price_int = int(vehicle_list[0].get("price_krw"))
        except Exception:
            price_int = None

        if price_int is not None and price_int > budget_max:
            def _fmt_manwon(val: int) -> str:
                man = val // 10_000
                return f"{man:,}만원"
            budget_str = _fmt_manwon(budget_max)
            price_str = _fmt_manwon(price_int)
            budget_warning_text = (
                f"사용자 메모 기준 예산 상한 {budget_str}보다 "
                f"추천 매물의 가격 {price_str}이 높습니다. "
                f"예산을 최우선으로 본다면 다른 매물을 보거나 가격을 재조정하는 것이 좋습니다."
            )


    # =========================
    # 3-A. 상단 공통 섹션 (단일/멀티 공통)
    #  - 요약 / 핵심 포인트 / 장단점 / 체크리스트 / 질문
    # =========================
    if is_multi:
        # 멀티일 때: best + ranking 구조 사용
        best = result.get("best") or {}

        summary = (
            result.get("summary")
            or result.get("summary_overall")
            or best.get("summary")
            or "요약 없음"
        )
        persona_label = result.get("persona_label", "")
        risk_level = best.get("risk_level") or result.get("risk_level", "")

        # --- 핵심 포인트 ---
        if isinstance(result.get("highlights"), list) and result["highlights"]:
            highlights = result["highlights"]
        elif isinstance(best.get("highlights"), list) and best["highlights"]:
            highlights = best["highlights"]
        elif best.get("summary"):
            highlights = [best["summary"]]
        else:
            highlights = []

        pros = best.get("pros", []) or []
        cons = best.get("cons", []) or []

        checklist = best.get("checklist")
        if not checklist:
            checklist = [
                "시동 후 공회전/주행 시 이상 소음·진동이 있는지 확인",
                "고속·저속 주행 시 핸들 떨림·쏠림 여부 확인",
                "사고·수리·정비 이력을 서류로 확인",
            ]

        questions = best.get("questions_for_seller", []) or []
        recommendation = result.get("recommendation", "")

    else:
        # 단일 매물: 그대로 result에서 직접 사용
        summary = result.get("summary", "요약 없음")
        persona_label = result.get("persona_label", "")
        risk_level = result.get("risk_level", "")
        highlights = result.get("highlights") or result.get("selling_points") or []
        pros = result.get("pros", [])
        cons = result.get("cons", [])
        checklist = result.get("checklist", [])
        questions = result.get("questions_for_seller", [])
        recommendation = result.get("recommendation", "")
        listing_text = result.get("listing_text", "")




        # --- 공통: 요약 + 캡션 ---
    st.markdown("#### 요약 (Summary)")
    st.write(summary if summary else "-")

    if persona_label or risk_level:
        mode_label = "차 사기 (구매)" if saved_mode == "buy" else "차 팔기 (판매)"
        caption = f"모드: {mode_label}"
        if persona_label:
            caption += f" | 페르소나: {persona_label}"
        if risk_level:
            caption += f" | 위험도: {risk_level}"
        st.caption(caption)
    # 💸 예산 경고 (buy 모드에서만)
    if saved_mode == "buy" and budget_warning_text:
        st.warning("💸 예산 체크: " + budget_warning_text)


    # =========================
    # 🔸 구매 모드 화면 (buy)
    # =========================
    if saved_mode == "buy":
        st.markdown("#### 핵심 포인트")
        if isinstance(highlights, list) and highlights:
            st.markdown("\n".join(f"- {h}" for h in highlights))
        elif isinstance(highlights, str) and highlights.strip():
            st.write(highlights)
        else:
            st.write("-")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 장점 (persona 기준)")
            if isinstance(pros, list) and pros:
                st.markdown("\n".join(f"- {p}" for p in pros))
            else:
                st.write("-")

        with col2:
            st.markdown("#### 단점 / 주의사항")
            if isinstance(cons, list) and cons:
                st.markdown("\n".join(f"- {c}" for c in cons))
            else:
                st.write("-")

        st.markdown("#### 시승/상담 체크리스트")
        if isinstance(checklist, list) and checklist:
            st.markdown("\n".join(f"- {c}" for c in checklist))
        else:
            st.write("-")

        st.markdown("#### 판매자/딜러에게 물어볼 질문")
        if isinstance(questions, list) and questions:
            st.markdown("\n".join(f"- {q}" for q in questions))
        else:
            st.write("-")

    # =========================
    # 🔹 판매 모드 화면 (sell)
    # =========================
    else:
        # 사이트에 올릴 문구 먼저 보여주기
        listing_title = result.get("listing_title", "")
        listing_body = result.get("listing_body", "")

        st.markdown("#### 사이트에 올릴 제목 (초안)")
        st.write(listing_title if listing_title else "-")

        st.markdown("#### 사이트에 올릴 설명 문구 (초안)")
        st.write(listing_body if listing_body else "-")

        st.markdown("---")

        st.markdown("#### 판매 시 강조하면 좋은 포인트")
        if isinstance(pros, list) and pros:
            st.markdown("\n".join(f"- {p}" for p in pros))
        else:
            st.write("-")

        st.markdown("#### 솔직하게 밝혀야 할 단점/주의사항")
        if isinstance(cons, list) and cons:
            st.markdown("\n".join(f"- {c}" for c in cons))
        else:
            st.write("-")

        st.markdown("#### 추천 판매 전략 / 코멘트")
        if isinstance(recommendation, str) and recommendation.strip():
            st.write(recommendation)
        else:
            st.write("-")






    # =========================
    # 3-B. 여러 매물일 때만 비교/랭킹 추가 표시
    # =========================
    if is_multi:
        ranking = result.get("ranking") or []

        if ranking:
            st.markdown("#### 여러 매물 우선순위")
            for rank_idx, item in enumerate(ranking, start=1):
                index = item.get("index", rank_idx)
                title = item.get("title") or f"{index}번 매물"
                fit_score = item.get("fit_score")
                score_txt = (
                    f"{float(fit_score):.1f}"
                    if isinstance(fit_score, (int, float))
                    else "-"
                )
                st.markdown(
                    f"- **#{rank_idx} 추천 매물** (원본 index: {index}, {title}) — "
                    f"적합도: {score_txt}/10.0"
                )

            # best_index 기준으로 최종 추천 강조 (가능하면 best 사용)
            best = result.get("best") or {}
            best_index = result.get("best_index", best.get("index", 1))
            try:
                best_index = int(best_index)
            except Exception:
                best_index = 1

            if not (1 <= best_index <= len(ranking)):
                best_index = 1

            best_title = best.get("title") or ranking[best_index - 1].get("title") or "제목 없음"
            st.success(f"✅ 최종 추천: #{best_index}번 매물 - {best_title}")
