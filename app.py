# app.py
# 🍽️ Travel Menu Recommender
# - Streamlit + OpenAI API
# - 메뉴 DB 우선 → 없으면 텍스트/이미지(OCR 선택) fallback
# - 추천 3개 + 구조화된 근거(룰 체인) + 피드백 누적(profile/history)

import os
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Optional OCR (no paid external services)
OCR_AVAILABLE = False
OCR_IMPORT_ERROR = ""
try:
    from PIL import Image  # pillow
    try:
        import pytesseract  # optional
        OCR_AVAILABLE = True
    except Exception as e:
        OCR_IMPORT_ERROR = f"pytesseract 불러오기 실패: {e}"
except Exception as e:
    OCR_IMPORT_ERROR = f"Pillow(PIL) 불러오기 실패: {e}"

# Optional OpenWeatherMap (manual weather is default)
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# OpenAI (latest-ish SDK, openai>=1.0)
OPENAI_AVAILABLE = True
OPENAI_IMPORT_ERROR = ""
try:
    from openai import OpenAI
except Exception as e:
    OPENAI_AVAILABLE = False
    OPENAI_IMPORT_ERROR = str(e)

# Asia/Seoul timezone (fixed offset)
KST = timezone(timedelta(hours=9))


# -----------------------------
# Sample Menu DB (small, embedded)
# -----------------------------
SAMPLE_MENU_DB = [
    {
        "restaurant_name": "Hanoi Street Eats",
        "country": "Vietnam",
        "city": "Hanoi",
        "menu_items": [
            {"name": "Phở Bò", "description": "소고기 쌀국수", "tags": ["국물", "담백", "따뜻함"], "price": 60000, "spice_level": 1},
            {"name": "Bún Chả", "description": "숯불 돼지고기 + 면", "tags": ["짭짤", "숯불", "든든"], "price": 70000, "spice_level": 1},
            {"name": "Gỏi Cuốn", "description": "월남쌈(생야채)", "tags": ["가벼움", "상큼", "건강식"], "price": 50000, "spice_level": 0},
            {"name": "Bánh Mì", "description": "바게트 샌드위치", "tags": ["바삭", "든든"], "price": 45000, "spice_level": 1},
        ],
    },
    {
        "restaurant_name": "Tokyo Cozy Diner",
        "country": "Japan",
        "city": "Tokyo",
        "menu_items": [
            {"name": "Shoyu Ramen", "description": "간장 라멘", "tags": ["국물", "짭짤", "따뜻함"], "price": 1200, "spice_level": 0},
            {"name": "Katsu Curry", "description": "돈카츠 카레", "tags": ["든든", "기름진", "따뜻함"], "price": 1400, "spice_level": 1},
            {"name": "Soba (Cold)", "description": "차가운 소바", "tags": ["담백", "시원함", "가벼움"], "price": 1100, "spice_level": 0},
            {"name": "Dorayaki", "description": "단팥 디저트", "tags": ["달콤", "디저트", "부드러움"], "price": 300, "spice_level": 0},
        ],
    },
    {
        "restaurant_name": "Barcelona Tapas Corner",
        "country": "Spain",
        "city": "Barcelona",
        "menu_items": [
            {"name": "Patatas Bravas", "description": "감자 + 매콤 소스", "tags": ["바삭", "매콤", "짭짤"], "price": 7.5, "spice_level": 3},
            {"name": "Gambas al Ajillo", "description": "마늘 새우", "tags": ["해산물", "향신료강함", "짭짤"], "price": 12.0, "spice_level": 1},
            {"name": "Pan con Tomate", "description": "토마토 빵", "tags": ["담백", "새콤", "가벼움"], "price": 5.0, "spice_level": 0},
            {"name": "Churros", "description": "디저트", "tags": ["달콤", "디저트", "바삭"], "price": 6.0, "spice_level": 0},
        ],
    },
]

DEFAULT_CURRENCIES = ["KRW", "USD", "EUR", "JPY", "VND", "THB", "SGD", "GBP", "AUD"]


# -----------------------------
# Utilities
# -----------------------------
def now_kst_iso() -> str:
    return datetime.now(tz=KST).isoformat(timespec="seconds")


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def init_state():
    if "menu_db" not in st.session_state:
        st.session_state.menu_db = SAMPLE_MENU_DB[:]  # copy-ish
    if "selected_restaurant_key" not in st.session_state:
        st.session_state.selected_restaurant_key = None
    if "menu_candidates" not in st.session_state:
        st.session_state.menu_candidates = []
    if "last_recommendations" not in st.session_state:
        st.session_state.last_recommendations = None
    if "profile" not in st.session_state:
        st.session_state.profile = {
            "preferred_tastes": {},
            "preferred_textures": {},
            "spice_preference": None,  # moving average
            "disliked_tags": {},
            "allergies": [],
        }
    if "history" not in st.session_state:
        st.session_state.history = []
    if "app_settings" not in st.session_state:
        st.session_state.app_settings = {
            "recommend_within_candidates_only": True,
            "use_openweather_if_key": False,
        }
    if "weather_mode" not in st.session_state:
        st.session_state.weather_mode = "manual"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def moving_average(prev: Optional[float], new: float, alpha: float = 0.2) -> float:
    if prev is None:
        return float(new)
    return float(prev) * (1 - alpha) + float(new) * alpha


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def simple_menu_parse(text: str) -> List[str]:
    """
    Very lightweight parser for pasted/OCR text:
    - split lines
    - drop empty/very short
    - remove obvious prices at end
    """
    items = []
    for raw in text.splitlines():
        line = normalize_text(raw)
        if len(line) < 2:
            continue
        # remove trailing prices like "12.0", "¥1200", "$7.5", "7,500", "70000"
        line = re.sub(r"[\s\-–—]*([₩¥$€£]?\s?\d[\d,]*(\.\d+)?)(\s*[A-Za-z]{0,3})?$", "", line).strip()
        if len(line) < 2:
            continue
        # remove bullet-like prefixes
        line = re.sub(r"^[•\-\*\d\.\)]\s*", "", line).strip()
        if len(line) < 2:
            continue
        items.append(line)
    # de-dup preserve order
    seen = set()
    out = []
    for i in items:
        key = i.lower()
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out[:60]


def auto_tag_menu_item(name: str, description: str = "") -> List[str]:
    """
    Simple heuristic tags from menu name/description for personalization updates.
    """
    t = (name + " " + (description or "")).lower()
    tags = []
    # textures
    if any(k in t for k in ["crispy", "fried", "katsu", "튀김", "바삭", "tempura"]):
        tags.append("바삭")
        tags.append("기름진")
    if any(k in t for k in ["soup", "ramen", "pho", "탕", "국", "국수", "broth", "stew", "찌개"]):
        tags.append("국물")
    if any(k in t for k in ["grill", "bbq", "숯", "구이"]):
        tags.append("숯불")
    if any(k in t for k in ["salad", "spring roll", "gỏi", "veggie", "야채", "채소"]):
        tags.append("건강식")
        tags.append("가벼움")
    if any(k in t for k in ["sweet", "dessert", "churros", "cake", "초코", "단팥", "푸딩"]):
        tags.append("달콤")
        tags.append("디저트")
    if any(k in t for k in ["spicy", "chili", "hot", "매운", "bravas", "kimchi"]):
        tags.append("매콤")
    if any(k in t for k in ["sour", "citrus", "vinegar", "tomato", "새콤"]):
        tags.append("새콤")
    if any(k in t for k in ["seafood", "shrimp", "fish", "해산물", "gambas"]):
        tags.append("해산물")
    return list(dict.fromkeys(tags))  # unique preserve order


def menu_items_from_db(restaurant: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for it in restaurant.get("menu_items", []):
        name = str(it.get("name", "")).strip()
        if not name:
            continue
        desc = it.get("description")
        tags = it.get("tags") or auto_tag_menu_item(name, desc or "")
        items.append(
            {
                "name": name,
                "description": desc,
                "tags": tags,
                "price": it.get("price"),
                "spice_level": it.get("spice_level"),
            }
        )
    return items


def menu_items_from_text(text: str) -> List[Dict[str, Any]]:
    names = simple_menu_parse(text)
    items = []
    for n in names:
        items.append({"name": n, "description": None, "tags": auto_tag_menu_item(n), "price": None, "spice_level": None})
    return items


def try_ocr_image(file) -> Tuple[Optional[str], str]:
    """
    Returns (text, note). If OCR unavailable or fails -> (None, reason).
    """
    if not OCR_AVAILABLE:
        return None, f"OCR 옵션을 사용할 수 없습니다. ({OCR_IMPORT_ERROR})\n텍스트 붙여넣기를 이용해 주세요."
    try:
        img = Image.open(file)
        # Basic OCR with English; often still extracts Latin menu names reasonably.
        text = pytesseract.image_to_string(img)
        text = text.strip()
        if len(text) < 5:
            return None, "OCR 결과가 너무 짧습니다. 텍스트 붙여넣기 입력으로 진행해 주세요."
        return text, "OCR 추출 성공"
    except Exception as e:
        return None, f"OCR 실패: {e}\n텍스트 붙여넣기 입력으로 진행해 주세요."


def openweather_fetch(city_country: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Minimal OpenWeatherMap fetch (optional). Requires requests.
    """
    if not REQUESTS_AVAILABLE:
        return None
    if not api_key or not city_country.strip():
        return None
    try:
        # OpenWeatherMap "q" can accept "city,country_code" but users may input free-form.
        # We'll try as-is.
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city_country, "appid": api_key, "units": "metric", "lang": "kr"}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        # normalize
        weather_main = (data.get("weather") or [{}])[0].get("main", "")
        weather_desc = (data.get("weather") or [{}])[0].get("description", "")
        temp = (data.get("main") or {}).get("temp", None)
        feels = (data.get("main") or {}).get("feels_like", None)
        humidity = (data.get("main") or {}).get("humidity", None)
        wind = (data.get("wind") or {}).get("speed", None)
        return {
            "provider": "OpenWeatherMap",
            "weather_main": weather_main,
            "weather_desc": weather_desc,
            "temp_c": temp,
            "feels_like_c": feels,
            "humidity": humidity,
            "wind_speed": wind,
            "raw": data,
        }
    except Exception:
        return None


def validate_minimum_inputs(inputs: Dict[str, Any]) -> Tuple[bool, str]:
    if not inputs.get("location"):
        return False, "현재 위치(도시/국가)를 입력해 주세요."
    if not inputs.get("weather_condition"):
        return False, "날씨를 선택해 주세요."
    if inputs.get("temperature_c") is None:
        return False, "온도를 설정해 주세요."
    if not inputs.get("condition"):
        return False, "컨디션을 선택해 주세요."
    if not inputs.get("meal_purpose"):
        return False, "식사 목적을 선택해 주세요."
    return True, ""


def profile_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    def top_k(d: Dict[str, float], k: int = 5):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

    return {
        "preferred_tastes_top": top_k(profile.get("preferred_tastes", {}), 6),
        "preferred_textures_top": top_k(profile.get("preferred_textures", {}), 6),
        "disliked_tags_top": top_k(profile.get("disliked_tags", {}), 6),
        "spice_preference_ma": profile.get("spice_preference", None),
        "allergies": profile.get("allergies", []),
    }


def update_profile_from_feedback(
    feedback_type: str,
    menu_item: Dict[str, Any],
    user_inputs: Dict[str, Any],
):
    """
    Update rules:
    - “좋았어요” -> +2
    - “별로였어요” -> -2 (as disliked_tags +2)
    - “먹었어요” -> +1
    Additionally update spice moving average from user's slider (since that's user truth).
    """
    if feedback_type not in ["ate", "like", "dislike"]:
        return

    delta = 0
    if feedback_type == "like":
        delta = 2
    elif feedback_type == "dislike":
        delta = -2
    else:
        delta = 1

    profile = st.session_state.profile
    tags = menu_item.get("tags") or []
    tastes_selected = user_inputs.get("taste_preferences") or []
    textures_selected = user_inputs.get("texture_preferences") or []

    # Update taste scores (use user's chosen taste signals as proxy)
    pt = profile.setdefault("preferred_tastes", {})
    if delta > 0:
        for t in tastes_selected:
            pt[t] = float(pt.get(t, 0.0) + delta)
    elif delta < 0:
        # negative feedback: slightly down-weight currently selected tastes (optional, mild)
        for t in tastes_selected:
            pt[t] = float(pt.get(t, 0.0) + delta * 0.5)

    # Update texture scores
    ptex = profile.setdefault("preferred_textures", {})
    if delta > 0:
        for tx in textures_selected:
            ptex[tx] = float(ptex.get(tx, 0.0) + delta)
    elif delta < 0:
        for tx in textures_selected:
            ptex[tx] = float(ptex.get(tx, 0.0) + delta * 0.5)

    # Update disliked tags
    dlt = profile.setdefault("disliked_tags", {})
    if feedback_type == "dislike":
        for tag in tags:
            dlt[tag] = float(dlt.get(tag, 0.0) + 2.0)  # accumulate disliked tag evidence
    elif feedback_type == "like":
        # if liked, reduce disliked score a bit for these tags
        for tag in tags:
            if tag in dlt:
                dlt[tag] = float(dlt.get(tag, 0.0) - 1.0)

    # spice moving average from user's preference slider (0~5)
    user_spice = float(user_inputs.get("spice_preference", 0))
    profile["spice_preference"] = moving_average(profile.get("spice_preference"), user_spice, alpha=0.25)

    # persist allergies in profile from sidebar (source of truth)
    profile["allergies"] = list(user_inputs.get("allergies", []))


def record_history_event(event: Dict[str, Any]):
    st.session_state.history.append(event)


def merge_or_overwrite_state(imported: Dict[str, Any], mode: str):
    """
    mode: 'merge' or 'overwrite'
    """
    if mode == "overwrite":
        st.session_state.profile = imported.get("profile", st.session_state.profile)
        st.session_state.history = imported.get("history", st.session_state.history)
        return

    # merge
    prof = st.session_state.profile
    imp_prof = imported.get("profile", {})

    for k in ["preferred_tastes", "preferred_textures", "disliked_tags"]:
        base = prof.setdefault(k, {})
        for kk, vv in (imp_prof.get(k, {}) or {}).items():
            base[kk] = float(base.get(kk, 0.0) + float(vv))

    # spice preference: average if both exist
    sp_a = prof.get("spice_preference", None)
    sp_b = imp_prof.get("spice_preference", None)
    if sp_b is not None:
        if sp_a is None:
            prof["spice_preference"] = float(sp_b)
        else:
            prof["spice_preference"] = float(sp_a) * 0.5 + float(sp_b) * 0.5

    # allergies: union
    a = set(prof.get("allergies", []) or [])
    b = set(imp_prof.get("allergies", []) or [])
    prof["allergies"] = sorted(list(a | b))

    # history: extend
    st.session_state.history.extend(imported.get("history", []) or [])


# -----------------------------
# OpenAI Call
# -----------------------------
RECOMMENDATION_JSON_SCHEMA = {
    "name": "travel_menu_recommendation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "meta": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "within_candidates_only": {"type": "boolean"},
                    "notes": {"type": "string"},
                },
                "required": ["within_candidates_only", "notes"],
            },
            "recommendations": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rank": {"type": "integer", "minimum": 1, "maximum": 3},
                        "menu_name": {"type": "string"},
                        "short_reason": {"type": "string"},
                        "structured_rationale": {
                            "type": "array",
                            "minItems": 2,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "signal": {"type": "string"},
                                    "context": {"type": "string"},
                                    "rule": {"type": "string"},
                                    "effect": {"type": "string"},
                                },
                                "required": ["signal", "context", "rule", "effect"],
                            },
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "cautions": {"type": "array", "items": {"type": "string"}},
                        "alternatives": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}},
                        "is_out_of_candidates": {"type": "boolean"},
                    },
                    "required": [
                        "rank",
                        "menu_name",
                        "short_reason",
                        "structured_rationale",
                        "confidence",
                        "cautions",
                        "alternatives",
                        "is_out_of_candidates",
                    ],
                },
            },
        },
        "required": ["meta", "recommendations"],
    },
}


def build_prompts(
    user_inputs: Dict[str, Any],
    menu_candidates: List[Dict[str, Any]],
    prof_summary: Dict[str, Any],
    within_candidates_only: bool,
) -> Tuple[str, str]:
    system = f"""
You are a senior travel food assistant that recommends what to eat right now while traveling.
You MUST output ONLY valid JSON that matches the provided JSON Schema (strict).
Safety/quality rules:
- Consider allergies/food restrictions and current condition (e.g., hangover, cold symptoms, upset stomach).
- Do NOT give medical advice or definitive health claims; use cautious language like "부담이 적을 수 있어요".
- Prefer gentle options when condition is poor (숙취/감기기운/속불편).
- If 'within_candidates_only' is true, you MUST recommend only from provided menu candidates.
- If you must suggest something outside candidates (only when within_candidates_only=false), mark is_out_of_candidates=true.
- Keep short_reason to 1–2 sentences.
- structured_rationale must be an if-then chain (array) with concrete signals and effects.
""".strip()

    # compact candidate list (avoid huge)
    candidates_compact = [
        {
            "name": it.get("name"),
            "description": it.get("description"),
            "tags": it.get("tags", []),
            "price": it.get("price"),
            "spice_level": it.get("spice_level"),
        }
        for it in menu_candidates[:60]
    ]

    user = {
        "task": "Recommend 3 menu items ranked 1..3.",
        "within_candidates_only": within_candidates_only,
        "inputs": user_inputs,
        "menu_candidates": candidates_compact,
        "profile_summary": prof_summary,
        "output_constraints": {
            "exactly_three_recommendations": True,
            "json_only": True,
        },
    }

    return system, safe_json_dumps(user)


def call_openai_recommendation(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not OPENAI_AVAILABLE:
        return None, f"openai 라이브러리를 불러올 수 없습니다: {OPENAI_IMPORT_ERROR}"
    if not api_key:
        return None, "OpenAI API Key가 없습니다."

    client = OpenAI(api_key=api_key)

    last_err = ""
    for attempt in range(1, max_retries + 2):
        try:
            # Responses API (recommended in newer SDK)
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_schema", "json_schema": RECOMMENDATION_JSON_SCHEMA},
                max_output_tokens=900,
            )
            text = (resp.output_text or "").strip()
            data = json.loads(text)
            return data, ""
        except Exception as e:
            last_err = f"[시도 {attempt}] {e}"
            # Fallback: try json_object (less strict) then validate lightly
            try:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_output_tokens=900,
                )
                text = (resp.output_text or "").strip()
                data = json.loads(text)
                # Minimal sanity check
                if isinstance(data, dict) and "recommendations" in data and len(data["recommendations"]) == 3:
                    # ensure required fields exist; if not, raise to retry
                    for r in data["recommendations"]:
                        for k in ["rank", "menu_name", "short_reason", "structured_rationale", "confidence", "cautions", "alternatives"]:
                            if k not in r:
                                raise ValueError("응답 JSON 필드 누락")
                        if "is_out_of_candidates" not in r:
                            r["is_out_of_candidates"] = False
                    if "meta" not in data:
                        data["meta"] = {"within_candidates_only": True, "notes": "json_object fallback"}
                    return data, ""
            except Exception:
                pass

    return None, f"추천 생성에 실패했습니다. 마지막 오류: {last_err}"


def openai_connection_test(api_key: str, model: str) -> Tuple[bool, str]:
    if not OPENAI_AVAILABLE:
        return False, f"openai 라이브러리 오류: {OPENAI_IMPORT_ERROR}"
    if not api_key:
        return False, "API Key가 비어있습니다."
    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": "ping"}],
            max_output_tokens=20,
        )
        _ = resp.output_text
        return True, "API 연결 성공"
    except Exception as e:
        return False, f"API 연결 실패: {e}"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="🍽️ Travel Menu Recommender", page_icon="🍽️", layout="wide")
init_state()

st.title("🍽️ Travel Menu Recommender")
st.caption("날씨/컨디션/취향을 반영해 지금 먹기 좋은 메뉴를 추천합니다.")

# Sidebar inputs (order must match requirements)
with st.sidebar:
    st.header("API 설정")
    api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("openai_api_key", ""))
    st.session_state.openai_api_key = api_key  # store but never print

    model = st.selectbox("모델 선택", ["gpt-4.1-mini", "gpt-4o-mini"], index=1)
    st.session_state.openai_model = model

    if st.button("API 연결 테스트"):
        with st.status("API 연결 테스트 중...", expanded=False) as s:
            ok, msg = openai_connection_test(api_key, model)
            if ok:
                s.update(label=msg, state="complete")
                st.toast("✅ API 연결 성공", icon="✅")
            else:
                s.update(label=msg, state="error")
                st.toast("❌ API 연결 실패", icon="❌")

    st.divider()

    st.header("여행/환경")
    location = st.text_input("현재 위치(도시/국가)", value=st.session_state.get("location", ""))
    st.session_state.location = location

    st.info("현재 위치 자동 가져오기(베타): Streamlit 웹앱은 브라우저 위치정보를 직접 받기 제약이 있어요. "
            "대신 도시/국가를 입력해 주세요.", icon="📍")

    # Weather: manual default, optional OpenWeatherMap
    st.subheader("날씨")
    owm_key = st.text_input("OpenWeatherMap API Key(선택)", type="password", value=st.session_state.get("owm_key", ""))
    st.session_state.owm_key = owm_key

    use_owm = st.toggle("OpenWeatherMap 연동 사용(키가 있을 때)", value=st.session_state.app_settings.get("use_openweather_if_key", False))
    st.session_state.app_settings["use_openweather_if_key"] = use_owm

    weather_condition = st.selectbox("날씨 선택", ["맑음", "비", "눈", "더움", "추움", "습함", "바람"])
    temperature_c = st.slider("온도(°C)", min_value=-10, max_value=40, value=int(st.session_state.get("temperature_c", 22)))

    st.session_state.weather_condition = weather_condition
    st.session_state.temperature_c = temperature_c

    fetched_weather = None
    if use_owm and owm_key and location:
        fetched_weather = openweather_fetch(location, owm_key)
        if fetched_weather:
            st.success(f"실제 날씨 연동됨: {fetched_weather.get('weather_desc','')} / {fetched_weather.get('temp_c')}°C", icon="🌦️")
        else:
            st.warning("실제 날씨를 가져오지 못해 수동 입력값을 사용합니다.", icon="⚠️")

    st.subheader("컨디션")
    condition = st.selectbox("컨디션 선택", ["아주좋음", "좋음", "보통", "안좋음", "숙취", "감기기운", "속불편"])
    activity = st.selectbox("활동량", ["낮음", "보통", "높음"])
    allergies = st.multiselect(
        "알레르기/금기(멀티선택)",
        ["유제품", "견과", "해산물", "글루텐", "돼지고기", "소고기", "채식", "할랄"],
        default=st.session_state.profile.get("allergies", []),
    )
    spice_preference = st.slider("맵기 선호(0~5)", min_value=0, max_value=5, value=int(st.session_state.get("spice_preference", 2)))

    st.session_state.condition = condition
    st.session_state.activity = activity
    st.session_state.allergies = allergies
    st.session_state.spice_preference = spice_preference
    # profile allergies always align with sidebar
    st.session_state.profile["allergies"] = allergies

    st.divider()

    st.header("취향")
    taste_preferences = st.multiselect(
        "맛 성향(멀티선택)",
        ["담백", "매콤", "달콤", "짭짤", "기름진", "새콤", "향신료강함"],
        default=st.session_state.get("taste_preferences", []),
    )
    texture_preferences = st.multiselect(
        "식감 선호",
        ["바삭", "부드러움", "쫄깃", "국물"],
        default=st.session_state.get("texture_preferences", []),
    )

    colb1, colb2 = st.columns([1, 1])
    with colb1:
        budget_value = st.number_input("1인 예산(숫자)", min_value=0.0, value=float(st.session_state.get("budget_value", 0.0)))
    with colb2:
        budget_currency = st.selectbox("통화", DEFAULT_CURRENCIES, index=DEFAULT_CURRENCIES.index(st.session_state.get("budget_currency", "KRW")) if st.session_state.get("budget_currency", "KRW") in DEFAULT_CURRENCIES else 0)

    meal_purpose = st.selectbox("식사 목적", ["든든한 한끼", "가벼운 한끼", "야식", "디저트", "해장", "건강식"])

    st.session_state.taste_preferences = taste_preferences
    st.session_state.texture_preferences = texture_preferences
    st.session_state.budget_value = budget_value
    st.session_state.budget_currency = budget_currency
    st.session_state.meal_purpose = meal_purpose

    st.divider()

    st.header("데이터 입력")

    # Restaurant DB search/select
    query = st.text_input("메뉴 DB에서 검색(restaurant_name / city)", value=st.session_state.get("restaurant_search", ""))
    st.session_state.restaurant_search = query

    def restaurant_key(r: Dict[str, Any]) -> str:
        return f"{r.get('restaurant_name')} | {r.get('city')}, {r.get('country')}"

    filtered = []
    ql = query.strip().lower()
    for r in st.session_state.menu_db:
        key = restaurant_key(r)
        if not ql or ql in key.lower():
            filtered.append(r)

    options = ["(선택 안 함)"] + [restaurant_key(r) for r in filtered]
    selected = st.selectbox("레스토랑 선택", options, index=0)
    selected_restaurant = None
    if selected != "(선택 안 함)":
        for r in filtered:
            if restaurant_key(r) == selected:
                selected_restaurant = r
                break

    if selected_restaurant:
        st.session_state.selected_restaurant_key = restaurant_key(selected_restaurant)
        with st.expander("선택한 레스토랑 메뉴 미리보기", expanded=False):
            items = menu_items_from_db(selected_restaurant)
            if items:
                st.dataframe(
                    [{"name": i["name"], "tags": ", ".join(i.get("tags", [])), "price": i.get("price"), "spice_level": i.get("spice_level")} for i in items],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("이 레스토랑에는 메뉴가 없습니다. 텍스트/사진 입력을 사용해 주세요.")
    else:
        st.session_state.selected_restaurant_key = None

    pasted_text = st.text_area("메뉴 텍스트 직접 붙여넣기(필수 백업)", value=st.session_state.get("pasted_menu_text", ""), height=120)
    st.session_state.pasted_menu_text = pasted_text

    uploaded_img = st.file_uploader("메뉴판 사진 업로드(jpg/png)", type=["jpg", "jpeg", "png"])
    st.session_state.uploaded_img_exists = uploaded_img is not None

    st.divider()
    st.subheader("추천 옵션")
    within_only = st.toggle(
        "후보 메뉴 안에서만 추천(기본 ON)",
        value=st.session_state.app_settings.get("recommend_within_candidates_only", True),
        help="ON이면 AI는 후보 리스트에 있는 메뉴만 추천합니다. OFF면 후보 외 추천도 가능(표기됨).",
    )
    st.session_state.app_settings["recommend_within_candidates_only"] = within_only

# Collect inputs dict for prompt
user_inputs = {
    "location": st.session_state.location,
    "weather_condition": st.session_state.weather_condition,
    "temperature_c": st.session_state.temperature_c,
    "weather_live": fetched_weather,  # may be None
    "condition": st.session_state.condition,
    "activity": st.session_state.activity,
    "allergies": st.session_state.allergies,
    "spice_preference": st.session_state.spice_preference,
    "taste_preferences": st.session_state.taste_preferences,
    "texture_preferences": st.session_state.texture_preferences,
    "budget": {"value": st.session_state.budget_value, "currency": st.session_state.budget_currency},
    "meal_purpose": st.session_state.meal_purpose,
    "selected_restaurant": st.session_state.selected_restaurant_key,
}

tab1, tab2, tab3 = st.tabs(["추천받기", "내 취향 기록(히스토리/피드백)", "설정/데이터(메뉴 DB 상태)"])


# -----------------------------
# Tab 1: Recommend
# -----------------------------
with tab1:
    st.subheader("추천받기")
    left, right = st.columns([1.2, 1])

    with right:
        st.markdown("#### 현재 입력 요약")
        st.write(
            {
                "위치": user_inputs["location"],
                "날씨": user_inputs["weather_condition"],
                "온도": f'{user_inputs["temperature_c"]}°C',
                "컨디션": user_inputs["condition"],
                "활동량": user_inputs["activity"],
                "식사 목적": user_inputs["meal_purpose"],
                "알레르기/금기": user_inputs["allergies"],
                "맵기 선호": user_inputs["spice_preference"],
                "맛 성향": user_inputs["taste_preferences"],
                "식감": user_inputs["texture_preferences"],
                "예산": f'{user_inputs["budget"]["value"]} {user_inputs["budget"]["currency"]}',
                "레스토랑": user_inputs["selected_restaurant"] or "(미선택)",
            }
        )

        ps = profile_summary(st.session_state.profile)
        with st.expander("내 취향 누적 요약(profile)", expanded=False):
            st.json(ps)

    with left:
        if st.button("메뉴 추천 받기", type="primary"):
            ok, msg = validate_minimum_inputs(user_inputs)
            if not ok:
                st.error(msg)
            else:
                # 1) Menu candidate 확보 (DB 우선)
                menu_candidates: List[Dict[str, Any]] = []
                source_note = ""

                if st.session_state.selected_restaurant_key:
                    # find restaurant in db by key
                    chosen = None
                    for r in st.session_state.menu_db:
                        if f"{r.get('restaurant_name')} | {r.get('city')}, {r.get('country')}" == st.session_state.selected_restaurant_key:
                            chosen = r
                            break
                    if chosen:
                        menu_candidates = menu_items_from_db(chosen)
                        source_note = "DB(선택 레스토랑)에서 메뉴 후보를 가져왔습니다."

                if not menu_candidates:
                    # 2) fallback: pasted text
                    if pasted_text.strip():
                        menu_candidates = menu_items_from_text(pasted_text)
                        source_note = "붙여넣기 텍스트에서 메뉴 후보를 생성했습니다."
                    else:
                        # 3) fallback: OCR from image (optional), then parse
                        if uploaded_img is not None:
                            with st.status("OCR로 메뉴판을 읽는 중(선택 옵션)...", expanded=True) as s:
                                text, note = try_ocr_image(uploaded_img)
                                st.write(note)
                                if text:
                                    st.code(text[:1200] + ("..." if len(text) > 1200 else ""))
                                    menu_candidates = menu_items_from_text(text)
                                    source_note = "OCR 결과에서 메뉴 후보를 생성했습니다."
                                    s.update(label="OCR 처리 완료", state="complete")
                                else:
                                    s.update(label="OCR 실패(텍스트 입력으로 대체 필요)", state="error")
                        if not menu_candidates:
                            st.error("메뉴 후보가 비어있습니다. DB 선택 / 메뉴 텍스트 붙여넣기 / 사진 업로드 중 하나를 제공해 주세요.")
                            st.stop()

                st.session_state.menu_candidates = menu_candidates

                # 3) OpenAI 호출 (strict JSON)
                if not api_key:
                    st.error("OpenAI API Key가 없습니다. 사이드바에서 입력해 주세요.")
                    st.stop()

                prof_sum = profile_summary(st.session_state.profile)
                system_prompt, user_prompt = build_prompts(
                    user_inputs=user_inputs,
                    menu_candidates=menu_candidates,
                    prof_summary=prof_sum,
                    within_candidates_only=within_only,
                )

                with st.spinner("AI가 메뉴를 추천하는 중..."):
                    rec, err = call_openai_recommendation(
                        api_key=api_key,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_retries=2,
                    )
                if err or rec is None:
                    st.error(err or "알 수 없는 오류")
                    st.stop()

                # 4) 후보 외 추천 처리(옵션에 따라 표시)
                candidate_names = {str(m.get("name", "")).strip() for m in menu_candidates}
                for r in rec.get("recommendations", []):
                    mn = str(r.get("menu_name", "")).strip()
                    if mn and (mn not in candidate_names):
                        r["is_out_of_candidates"] = True
                    else:
                        r["is_out_of_candidates"] = False

                st.session_state.last_recommendations = rec

                # 5) history 기록(추천 라운드)
                round_event = {
                    "timestamp": now_kst_iso(),
                    "location": user_inputs["location"],
                    "weather_condition": user_inputs["weather_condition"],
                    "temperature_c": user_inputs["temperature_c"],
                    "condition": user_inputs["condition"],
                    "activity": user_inputs["activity"],
                    "meal_purpose": user_inputs["meal_purpose"],
                    "budget": user_inputs["budget"],
                    "selected_restaurant": user_inputs["selected_restaurant"],
                    "menu_source": source_note,
                    "candidates_count": len(menu_candidates),
                    "recommendations": rec.get("recommendations", []),
                    "feedback": {},  # filled by buttons later
                }
                record_history_event(round_event)
                st.toast("✅ 추천 생성 완료", icon="✅")

        # Display recommendations if exist
        rec = st.session_state.last_recommendations
        if rec:
            st.info(rec.get("meta", {}).get("notes", "추천 결과"), icon="ℹ️")

            # Split within/outside candidates for clarity
            in_cand = [r for r in rec.get("recommendations", []) if not r.get("is_out_of_candidates")]
            out_cand = [r for r in rec.get("recommendations", []) if r.get("is_out_of_candidates")]

            def render_card(r: Dict[str, Any], idx: int):
                rank = r.get("rank", idx + 1)
                conf = r.get("confidence", 0.5)
                menu_name = r.get("menu_name", "")
                short_reason = r.get("short_reason", "")
                cautions = r.get("cautions", []) or []
                alternatives = r.get("alternatives", []) or []
                rationale = r.get("structured_rationale", []) or []

                badge = f"🏅 Rank {rank}"
                conf_pct = int(clamp01(float(conf)) * 100)

                container = st.container(border=True)
                with container:
                    topc1, topc2 = st.columns([1, 1])
                    with topc1:
                        st.markdown(f"### {badge} — **{menu_name}**")
                    with topc2:
                        st.metric("Confidence", f"{conf_pct}%")

                    if r.get("is_out_of_candidates"):
                        st.warning("⚠️ 메뉴 후보 외 추천(대안)입니다.", icon="⚠️")

                    st.write(short_reason)

                    if cautions:
                        st.error("주의사항: " + " / ".join([str(x) for x in cautions]), icon="🚧")

                    if alternatives:
                        st.caption("비슷한 메뉴(Alternatives): " + ", ".join([str(x) for x in alternatives]))

                    with st.expander("추천 근거(구조화: 룰 체인) 보기", expanded=False):
                        rows = []
                        for rr in rationale:
                            rows.append(
                                {
                                    "signal": rr.get("signal", ""),
                                    "context": rr.get("context", ""),
                                    "rule": rr.get("rule", ""),
                                    "effect": rr.get("effect", ""),
                                }
                            )
                        if rows:
                            st.dataframe(rows, use_container_width=True, hide_index=True)
                        else:
                            st.write("근거 데이터가 없습니다.")

                    # Feedback buttons
                    b1, b2, b3 = st.columns(3)
                    key_base = f"fb_{now_kst_iso()}_{rank}_{idx}"
                    with b1:
                        if st.button("이거 먹었어요 👍", key=key_base + "_ate"):
                            apply_feedback(r, "ate", user_inputs)
                    with b2:
                        if st.button("좋았어요 😊", key=key_base + "_like"):
                            apply_feedback(r, "like", user_inputs)
                    with b3:
                        if st.button("별로였어요 😕", key=key_base + "_dislike"):
                            apply_feedback(r, "dislike", user_inputs)

            def find_menu_item_by_name(name: str) -> Dict[str, Any]:
                for m in st.session_state.menu_candidates:
                    if str(m.get("name", "")).strip() == str(name).strip():
                        return m
                # if out-of-candidates: minimal object
                return {"name": name, "description": None, "tags": auto_tag_menu_item(name), "price": None, "spice_level": None}

            def apply_feedback(rec_item: Dict[str, Any], feedback_type: str, inputs: Dict[str, Any]):
                # Update profile
                menu_item = find_menu_item_by_name(rec_item.get("menu_name", ""))
                update_profile_from_feedback(feedback_type, menu_item, inputs)

                # Log to latest history event
                if st.session_state.history:
                    st.session_state.history[-1].setdefault("feedback", {})
                    st.session_state.history[-1]["feedback"][rec_item.get("menu_name", "")] = feedback_type

                st.toast("✅ 피드백이 반영되었습니다. 다음 추천부터 더 정확해져요!", icon="✅")

            # Render cards
            if in_cand:
                st.markdown("#### 추천 메뉴(후보 내)")
                for i, r in enumerate(sorted(in_cand, key=lambda x: x.get("rank", 99))):
                    render_card(r, i)

            if out_cand:
                st.markdown("#### 메뉴 후보 외 추천(대안)")
                for i, r in enumerate(sorted(out_cand, key=lambda x: x.get("rank", 99))):
                    render_card(r, i + 10)


# -----------------------------
# Tab 2: History / Profile
# -----------------------------
with tab2:
    st.subheader("내 취향 기록")

    hist = st.session_state.history
    if not hist:
        st.info("아직 추천/피드백 기록이 없습니다. 탭1에서 추천을 받아보세요.", icon="📝")
    else:
        rows = []
        for h in hist[-50:][::-1]:
            rec_names = [r.get("menu_name", "") for r in (h.get("recommendations") or [])]
            fb = h.get("feedback", {}) or {}
            chosen = [k for k, v in fb.items() if v in ["ate", "like", "dislike"]]
            rows.append(
                {
                    "날짜(KST)": h.get("timestamp"),
                    "도시/국가": h.get("location"),
                    "날씨": f'{h.get("weather_condition")} / {h.get("temperature_c")}°C',
                    "추천메뉴": ", ".join([x for x in rec_names if x]),
                    "선택/피드백": ", ".join([f"{k}:{fb.get(k)}" for k in chosen]) if chosen else "",
                    "메뉴소스": h.get("menu_source", ""),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("### 내 취향 요약")
    ps = profile_summary(st.session_state.profile)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**자주 좋아한 맛 성향 TOP**")
        if ps["preferred_tastes_top"]:
            st.write(ps["preferred_tastes_top"])
        else:
            st.caption("아직 데이터가 부족합니다.")
    with c2:
        st.markdown("**자주 좋아한 식감 TOP**")
        if ps["preferred_textures_top"]:
            st.write(ps["preferred_textures_top"])
        else:
            st.caption("아직 데이터가 부족합니다.")
    with c3:
        st.markdown("**싫어요 패턴(태그) TOP**")
        if ps["disliked_tags_top"]:
            st.write(ps["disliked_tags_top"])
        else:
            st.caption("아직 데이터가 부족합니다.")

    if ps["spice_preference_ma"] is not None:
        st.info(f"누적 맵기 선호(이동평균): **{ps['spice_preference_ma']:.2f} / 5**", icon="🌶️")

    st.divider()
    st.markdown("### 데이터 초기화")
    confirm = st.checkbox("정말 초기화할게요(확인)")
    if st.button("전체 데이터 삭제(초기화)", disabled=not confirm):
        st.session_state.profile = {
            "preferred_tastes": {},
            "preferred_textures": {},
            "spice_preference": None,
            "disliked_tags": {},
            "allergies": st.session_state.allergies,
        }
        st.session_state.history = []
        st.session_state.last_recommendations = None
        st.toast("🧹 초기화 완료", icon="🧹")
        st.rerun()


# -----------------------------
# Tab 3: Settings / Data
# -----------------------------
with tab3:
    st.subheader("설정/데이터")

    st.markdown("### 메뉴 DB 상태")
    st.caption("샘플 DB가 내장되어 있으며, JSON 업로드로 확장할 수 있습니다.")
    st.write(f"현재 레스토랑 수: **{len(st.session_state.menu_db)}**")

    with st.expander("현재 메뉴 DB 미리보기", expanded=False):
        preview_rows = []
        for r in st.session_state.menu_db[:30]:
            preview_rows.append(
                {
                    "restaurant_name": r.get("restaurant_name"),
                    "country": r.get("country"),
                    "city": r.get("city"),
                    "menu_items_count": len(r.get("menu_items", []) or []),
                }
            )
        st.dataframe(preview_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 메뉴 DB JSON 업로드(확장)")
    st.caption("형식: restaurant_name, country, city, menu_items[{name, description(optional), tags, price(optional), spice_level(optional)}]")

    db_upload = st.file_uploader("메뉴 DB JSON 업로드", type=["json"], key="db_upload")
    if db_upload is not None:
        try:
            imported = json.loads(db_upload.read().decode("utf-8"))
            if isinstance(imported, dict) and "restaurants" in imported:
                imported = imported["restaurants"]
            if not isinstance(imported, list):
                raise ValueError("JSON은 레스토랑 객체의 배열(list)이어야 합니다. (또는 {'restaurants':[...]} 형태)")

            # simple validation
            valid = []
            for r in imported:
                if not isinstance(r, dict):
                    continue
                if not r.get("restaurant_name") or not r.get("city") or not r.get("country"):
                    continue
                if "menu_items" not in r:
                    r["menu_items"] = []
                valid.append(r)

            if st.button("DB에 병합 추가"):
                st.session_state.menu_db.extend(valid)
                st.toast(f"✅ DB 병합 완료: +{len(valid)}개 레스토랑", icon="✅")
                st.rerun()
        except Exception as e:
            st.error(f"DB JSON 처리 실패: {e}")

    st.divider()
    st.markdown("### 내 취향/히스토리 내보내기")
    export_obj = {"profile": st.session_state.profile, "history": st.session_state.history}
    export_json = safe_json_dumps(export_obj)
    st.download_button(
        "JSON 다운로드",
        data=export_json.encode("utf-8"),
        file_name="travel_menu_profile_history.json",
        mime="application/json",
    )

    st.markdown("### 내 취향/히스토리 가져오기")
    imp = st.file_uploader("JSON 가져오기", type=["json"], key="pref_import")
    import_mode = st.radio("가져오기 방식", ["병합(merge)", "덮어쓰기(overwrite)"], horizontal=True)
    if imp is not None:
        try:
            imported = json.loads(imp.read().decode("utf-8"))
            if not isinstance(imported, dict) or ("profile" not in imported and "history" not in imported):
                raise ValueError("올바른 형식이 아닙니다. {'profile':..., 'history':...} 형태여야 합니다.")
            mode = "merge" if import_mode.startswith("병합") else "overwrite"
            if st.button("가져오기 적용"):
                merge_or_overwrite_state(imported, mode)
                st.toast("✅ 가져오기 완료", icon="✅")
                st.rerun()
        except Exception as e:
            st.error(f"가져오기 실패: {e}")

    st.divider()
    st.markdown("### 로컬 저장 옵션(서버 파일로 저장)")
    st.caption("Streamlit이 실행되는 서버/로컬 환경에 JSON 파일로 저장합니다. (배포 환경에선 권한/경로 제한이 있을 수 있어요.)")
    save_name = st.text_input("저장 파일명", value="local_travel_menu_state.json")
    csave1, csave2 = st.columns([1, 1])
    with csave1:
        if st.button("로컬 파일로 저장"):
            try:
                with open(save_name, "w", encoding="utf-8") as f:
                    f.write(export_json)
                st.success(f"저장 완료: {save_name}")
            except Exception as e:
                st.error(f"저장 실패: {e}")
    with csave2:
        if st.button("로컬 파일에서 불러오기"):
            try:
                with open(save_name, "r", encoding="utf-8") as f:
                    imported = json.loads(f.read())
                merge_or_overwrite_state(imported, "overwrite")
                st.success(f"불러오기 완료: {save_name}")
                st.rerun()
            except Exception as e:
                st.error(f"불러오기 실패: {e}")

# Footer: OCR note
with st.sidebar:
    st.divider()
    st.caption("OCR(선택): pytesseract + tesseract 설치 시 사진에서 텍스트 추출이 가능합니다.")
    if not OCR_AVAILABLE:
        st.caption(f"현재 OCR 비활성: {OCR_IMPORT_ERROR}")


# -----------------------------
# Run guide (in-app)
# -----------------------------
with st.expander("실행 방법", expanded=False):
    st.markdown(
        """
1) 설치
```bash
pip install streamlit openai pillow
# (선택) OCR 사용 시:
pip install pytesseract
# 그리고 OS에 tesseract 설치가 필요합니다 (예: macOS brew install tesseract)
