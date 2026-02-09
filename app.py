import os
import json
import sqlite3
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# OpenAI (Python SDK)
from openai import OpenAI

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="🍽️ 메뉴 추천 (여행용)", page_icon="🍽️", layout="wide")

DEFAULT_MODEL = "gpt-5-mini"  # 필요하면 사이드바에서 바꿀 수 있게 해둠

# -----------------------------
# DB (피드백 누적)
# -----------------------------
@st.cache_resource
def get_conn(db_path: str = "menu_feedback.sqlite3") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            user_id TEXT NOT NULL,
            country TEXT,
            city TEXT,
            restaurant TEXT,
            recommended_menu TEXT,
            chosen_menu TEXT,
            sentiment TEXT, -- "like" | "dislike" | "neutral"
            notes TEXT,
            context_json TEXT
        )
        """
    )
    conn.commit()
    return conn

def save_feedback(
    conn: sqlite3.Connection,
    row: Dict[str, Any],
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback (
            created_at, user_id, country, city, restaurant,
            recommended_menu, chosen_menu, sentiment, notes, context_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("created_at"),
            row.get("user_id"),
            row.get("country"),
            row.get("city"),
            row.get("restaurant"),
            row.get("recommended_menu"),
            row.get("chosen_menu"),
            row.get("sentiment"),
            row.get("notes"),
            json.dumps(row.get("context_json", {}), ensure_ascii=False),
        ),
    )
    conn.commit()

def load_recent_feedback(conn: sqlite3.Connection, user_id: str, limit: int = 50) -> pd.DataFrame:
    q = """
    SELECT created_at, country, city, restaurant, recommended_menu, chosen_menu, sentiment, notes, context_json
    FROM feedback
    WHERE user_id = ?
    ORDER BY id DESC
    LIMIT ?
    """
    df = pd.read_sql_query(q, conn, params=(user_id, limit))
    return df

def summarize_taste_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    매우 단순한 누적 프로필 예시:
    - like/dislike 빈도
    - chosen_menu/notes 키워드 기반 힌트(가벼운 수준)
    실제 서비스에서는 임베딩/클러스터링/랭킹으로 고도화 추천.
    """
    if df.empty:
        return {"history": "없음", "likes": [], "dislikes": [], "stats": {}}

    likes = df[df["sentiment"] == "like"]["chosen_menu"].dropna().tolist()
    dislikes = df[df["sentiment"] == "dislike"]["chosen_menu"].dropna().tolist()

    stats = {
        "total": len(df),
        "like": int((df["sentiment"] == "like").sum()),
        "dislike": int((df["sentiment"] == "dislike").sum()),
        "neutral": int((df["sentiment"] == "neutral").sum()),
    }
    return {
        "history": "최근 피드백 기반",
        "likes": likes[:10],
        "dislikes": dislikes[:10],
        "stats": stats,
    }

# -----------------------------
# 위치/날씨
# -----------------------------
def get_location_from_browser() -> Optional[Dict[str, float]]:
    """
    브라우저 geolocation API를 Streamlit component로 호출.
    사용자가 위치 권한을 허용해야 함.
    """
    js = """
    <script>
    const sendLocation = () => {
      if (!navigator.geolocation) {
        Streamlit.setComponentValue({error: "Geolocation not supported"});
        return;
      }
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          Streamlit.setComponentValue({
            lat: pos.coords.latitude,
            lon: pos.coords.longitude
          });
        },
        (err) => {
          Streamlit.setComponentValue({error: err.message});
        }
      );
    };
    sendLocation();
    </script>
    """
    result = components.html(js, height=0)
    return result

def reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
    """
    Nominatim(OpenStreetMap) reverse geocoding (무료/무키)
    - 상용/대규모 트래픽은 정책 준수 필요
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "jsonv2", "lat": lat, "lon": lon}
    headers = {"User-Agent": "menu-reco-app/1.0 (educational)"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    addr = data.get("address", {})
    return {
        "country": addr.get("country"),
        "city": addr.get("city") or addr.get("town") or addr.get("village") or addr.get("state"),
        "display_name": data.get("display_name"),
    }

def fetch_weather_open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    """
    Open-Meteo (무료/무키) 현재 날씨
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    cur = data.get("current", {})
    return {
        "temp_c": cur.get("temperature_2m"),
        "feels_like_c": cur.get("apparent_temperature"),
        "humidity": cur.get("relative_humidity_2m"),
        "precip_mm": cur.get("precipitation"),
        "wind_kmh": cur.get("wind_speed_10m"),
        "weather_code": cur.get("weather_code"),
        "time": cur.get("time"),
    }

# -----------------------------
# 메뉴 DB 업로드/정리
# -----------------------------
def normalize_menu_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    최소 컬럼:
    - country, city, restaurant, menu_name, description(optional), tags(optional), price(optional)
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["country", "city", "restaurant", "menu_name"]
    for c in required:
        if c not in df.columns:
            df[c] = ""

    for opt in ["description", "tags", "price"]:
        if opt not in df.columns:
            df[opt] = ""

    return df[["country", "city", "restaurant", "menu_name", "description", "tags", "price"]]

def filter_menu_df(df: pd.DataFrame, country: str, city: str, restaurant: str = "") -> pd.DataFrame:
    f = df.copy()
    if country:
        f = f[f["country"].str.contains(country, case=False, na=False)]
    if city:
        f = f[f["city"].str.contains(city, case=False, na=False)]
    if restaurant:
        f = f[f["restaurant"].str.contains(restaurant, case=False, na=False)]
    return f

# -----------------------------
# OpenAI helpers
# -----------------------------
def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def oai_extract_menu_from_image(
    client: OpenAI,
    model: str,
    image_bytes: bytes,
    hint_locale: str = "ko",
) -> Dict[str, Any]:
    """
    메뉴판 사진 -> 구조화된 메뉴 리스트 추출
    Responses API는 이미지 입력을 지원. :contentReference[oaicite:1]{index=1}
    """
    prompt = f"""
너는 메뉴판을 구조화하는 파서야.
메뉴판 이미지에서 다음 JSON 스키마로만 추출해.
언어는 가능하면 {hint_locale}로 정리하고, 원문이 영어/현지어면 원문도 함께 보존해.

출력 JSON 스키마:
{{
  "restaurant": "추정 식당명(없으면 빈 문자열)",
  "currency": "통화기호(알 수 없으면 빈 문자열)",
  "items": [
    {{
      "menu_name": "메뉴명",
      "description": "설명(없으면 빈 문자열)",
      "price": "가격(없으면 빈 문자열)",
      "tags": ["재료/조리법/특징 키워드들(추론 가능)"]
    }}
  ]
}}

주의:
- JSON 외 텍스트 출력 금지
- items는 최소 5개 이상이면 좋고, 없으면 빈 배열
"""
    # Responses API: input에 텍스트 + 이미지
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_bytes": image_bytes},
                ],
            }
        ],
    )
    text = resp.output_text
    try:
        return json.loads(text)
    except Exception:
        # 모델이 JSON을 깨면, 최소한의 복구 시도
        return {"restaurant": "", "currency": "", "items": [], "raw": text}

def oai_recommend_menu(
    client: OpenAI,
    model: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    최종 추천: 메뉴 후보 + 근거(구조적 rules) + 신뢰 설명
    """
    system = """
너는 여행 중 메뉴 추천 전문가야.
중요: 사용자가 추천을 "왜 신뢰해야 하는지"를 구조적으로 보여줘야 해.
추론은 하되, 반드시 사용자가 준 정보/메뉴 데이터에 근거를 연결해.

출력은 JSON만. JSON 외 텍스트 금지.

출력 스키마:
{
  "top_picks": [
    {
      "menu_name": "...",
      "restaurant": "...",
      "why": ["짧은 이유 1", "짧은 이유 2"],
      "evidence_rules": [
        {"if": "사용자 조건", "then": "추천 논리", "because": "메뉴/날씨/피드백 근거"},
        ...
      ],
      "confidence": 0.0,
      "cautions": ["주의사항(알레르기/매움/날씨 등)"]
    }
  ],
  "trust_explainer": {
    "data_used": ["menu_db" | "menu_photo" | "user_history" | "weather" | "location"],
    "limitations": ["이 추천의 한계/불확실성"]
  },
  "follow_up_questions": ["정확도를 위해 물어볼 1~3개 질문"]
}

규칙:
- top_picks는 3개
- confidence는 0~1
- evidence_rules는 최소 3개/추천
- 메뉴가 기름지다/담백하다 같은 속성 추정 시: 'because'에 '추정'이라고 명시
"""
    user = json.dumps(context, ensure_ascii=False)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user}]},
        ],
    )

    out = resp.output_text
    try:
        return json.loads(out)
    except Exception:
        return {"error": "Model returned invalid JSON", "raw": out, "context": context}

# -----------------------------
# UI
# -----------------------------
st.title("🍽️ 여행지 맞춤 메뉴 추천")
st.caption("현재 위치/날씨/컨디션/취향 + (메뉴 DB 또는 메뉴판 사진) 기반으로 추천하고, 추천 근거를 규칙 형태로 보여줍니다.")

with st.sidebar:
    st.header("🔐 OpenAI 설정")
    api_key = st.text_input("OpenAI API Key", type="password", help="키는 로컬에 저장하지 않는 것을 권장합니다.")
    model = st.text_input("Model", value=DEFAULT_MODEL, help="예: gpt-5-mini 등")
    user_id = st.text_input("사용자 ID", value="demo_user", help="피드백 누적용 식별자(닉네임 등)")

    st.divider()
    st.header("📍 위치/날씨")
    loc_mode = st.radio("위치 입력 방식", ["브라우저에서 자동 가져오기", "직접 입력"], index=0)

    lat = lon = None
    country = city = ""
    location_display = ""

    if loc_mode == "브라우저에서 자동 가져오기":
        if st.button("현재 위치 가져오기"):
            result = get_location_from_browser()
            if isinstance(result, dict) and result.get("error"):
                st.error(f"위치 가져오기 실패: {result['error']}")
            elif isinstance(result, dict) and "lat" in result and "lon" in result:
                lat, lon = float(result["lat"]), float(result["lon"])
                st.success(f"좌표: {lat:.5f}, {lon:.5f}")
                try:
                    geo = reverse_geocode(lat, lon)
                    country = geo.get("country") or ""
                    city = geo.get("city") or ""
                    location_display = geo.get("display_name") or ""
                    st.write(f"추정 위치: {country} / {city}")
                except Exception as e:
                    st.warning(f"주소 변환 실패(좌표는 사용 가능): {e}")
            else:
                st.info("브라우저 권한 허용 후 다시 시도해 주세요.")
    else:
        country = st.text_input("국가", value="")
        city = st.text_input("도시", value="")
        lat = st.number_input("위도(lat)", value=0.0, format="%.6f")
        lon = st.number_input("경도(lon)", value=0.0, format="%.6f")

    want_weather = st.toggle("날씨 자동 조회", value=True)
    weather = {}
    if want_weather and lat is not None and lon is not None and (lat != 0.0 or lon != 0.0):
        if st.button("날씨 조회"):
            try:
                weather = fetch_weather_open_meteo(lat, lon)
                st.success(f"현재 {weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C)")
            except Exception as e:
                st.error(f"날씨 조회 실패: {e}")

    st.divider()
    st.header("🧾 메뉴 데이터")
    st.caption("1) 메뉴 DB 업로드가 우선. 없으면 2) 메뉴판 사진 업로드.")
    menu_file = st.file_uploader("메뉴 DB 파일 (CSV/JSON)", type=["csv", "json"])
    menu_photo = st.file_uploader("메뉴판 사진 (대체 입력)", type=["png", "jpg", "jpeg"])

# 본문 입력
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🙋 사용자 상태/취향")
    condition = st.selectbox("오늘 컨디션", ["정상", "피곤함", "숙취", "감기기운", "속이 더부룩함", "운동 후", "스트레스"], index=0)
    hunger = st.select_slider("배고픔 정도", options=["조금", "보통", "많이"], value="보통")
    diet = st.multiselect("식단/제약", ["채식", "비건", "할랄", "글루텐프리", "유제품 피하기", "해산물 피하기", "돼지고기 피하기"], default=[])
    allergies = st.text_input("알레르기/못 먹는 재료(자유입력)", value="", placeholder="예: 땅콩, 갑각류, 고수 등")

    taste = st.multiselect(
        "음식 취향(복수 선택)",
        ["담백한", "매운", "기름진", "국물", "면", "밥", "고기", "해산물", "채소", "달달한", "새콤한", "향신료 강한", "치즈/크리미"],
        default=["담백한"],
    )
    adventurous = st.slider("현지 음식 도전 의향", 0, 10, 6)
    budget = st.selectbox("예산", ["저렴", "중간", "상관없음"], index=1)

with col2:
    st.subheader("🏪 레스토랑/상황")
    restaurant = st.text_input("식당 이름(알면)", value="", placeholder="모르면 비워도 됨")
    dining = st.selectbox("식사 상황", ["혼밥", "친구/연인과", "가족", "비즈니스"], index=0)
    time_of_day = st.selectbox("시간대", ["아침", "점심", "저녁", "야식"], index=2)
    note = st.text_area("추가 요청(자유)", value="", placeholder="예: 속이 안 좋아서 자극적인 건 피하고 싶어 / 더운 날이라 시원한 거 등")

st.divider()

# 피드백/히스토리
conn = get_conn()
df_hist = load_recent_feedback(conn, user_id=user_id, limit=50)
profile = summarize_taste_profile(df_hist)

with st.expander("📚 내 취향 히스토리(누적 피드백)", expanded=False):
    st.write(profile["stats"])
    if profile["likes"]:
        st.success("최근 좋았던 메뉴: " + ", ".join(profile["likes"][:8]))
    if profile["dislikes"]:
        st.error("최근 별로였던 메뉴: " + ", ".join(profile["dislikes"][:8]))
    if not df_hist.empty:
        st.dataframe(df_hist, use_container_width=True)

# 메뉴 후보 만들기
menu_source = None
menu_items: List[Dict[str, Any]] = []
menu_meta: Dict[str, Any] = {}

if menu_file is not None:
    try:
        if menu_file.name.lower().endswith(".csv"):
            df = pd.read_csv(menu_file)
        else:
            df = pd.read_json(menu_file)
        df = normalize_menu_df(df)
        df_f = filter_menu_df(df, country=country, city=city, restaurant=restaurant)
        if not df_f.empty:
            menu_source = "menu_db"
            menu_items = df_f.to_dict(orient="records")
            menu_meta = {"rows": len(df_f), "country": country, "city": city, "restaurant_filter": restaurant}
        else:
            # 업로드는 했지만 필터 결과가 없는 경우에도 전체를 쓰게 할지 선택 가능
            menu_source = "menu_db"
            menu_items = df.to_dict(orient="records")[:300]
            menu_meta = {"rows": len(df), "note": "필터 결과가 없어 전체 DB 일부(최대 300행) 사용"}
    except Exception as e:
        st.warning(f"메뉴 DB 파싱 실패: {e}")

# 메뉴판 사진에서 추출
if (not menu_items) and (menu_photo is not None) and api_key:
    st.info("메뉴 DB가 없거나 비어 있어, 메뉴판 사진에서 메뉴를 추출합니다.")
    try:
        client = make_client(api_key)
        extracted = oai_extract_menu_from_image(client, model=model, image_bytes=menu_photo.getvalue(), hint_locale="ko")
        items = extracted.get("items", []) or []
        if items:
            menu_source = "menu_photo"
            # DB와 맞추기 위해 형태 통일
            for it in items:
                menu_items.append({
                    "country": country,
                    "city": city,
                    "restaurant": extracted.get("restaurant", "") or restaurant,
                    "menu_name": it.get("menu_name", ""),
                    "description": it.get("description", ""),
                    "tags": ",".join(it.get("tags", []) or []),
                    "price": it.get("price", ""),
                })
            menu_meta = {"restaurant": extracted.get("restaurant", ""), "currency": extracted.get("currency", ""), "items": len(menu_items)}
        else:
            st.warning("사진에서 메뉴를 충분히 추출하지 못했습니다. 더 선명한 사진을 올려주세요.")
    except Exception as e:
        st.error(f"메뉴판 분석 실패: {e}")

# 추천 실행
st.subheader("✨ 추천 받기")

disabled = (not api_key)
if disabled:
    st.warning("사이드바에 OpenAI API Key를 입력해야 추천을 생성할 수 있어요.")

run = st.button("🍜 메뉴 추천 생성", type="primary", disabled=disabled)

if run:
    client = make_client(api_key)

    # 날씨 없으면, 그래도 동작하도록 빈 값 허용
    weather_used = weather if isinstance(weather, dict) and weather else {}

    context = {
        "location": {
            "country": country,
            "city": city,
            "restaurant": restaurant,
            "lat": lat,
            "lon": lon,
            "display_name": location_display,
        },
        "weather": weather_used,
        "user_state": {
            "condition": condition,
            "hunger": hunger,
            "diet": diet,
            "allergies": allergies,
            "taste": taste,
            "adventurous": adventurous,
            "budget": budget,
            "dining": dining,
            "time_of_day": time_of_day,
            "extra_note": note,
        },
        "history_profile": profile,
        "menu_source": menu_source or "none",
        "menu_meta": menu_meta,
        # 모델에 전부 다 주면 길어질 수 있어서 상한
        "menu_candidates": menu_items[:250],
        "output_requirements": {
            "show_structured_evidence_rules": True,
            "need_trust_explainer": True,
            "need_3_picks": True,
        },
    }

    with st.spinner("AI가 메뉴를 고르는 중..."):
        result = oai_recommend_menu(client, model=model, context=context)

    if result.get("error"):
        st.error(result["error"])
        st.code(result.get("raw", ""))
    else:
        picks = result.get("top_picks", [])
        trust = result.get("trust_explainer", {})
        follow = result.get("follow_up_questions", [])

        st.markdown("### ✅ 추천 결과")
        for i, p in enumerate(picks, start=1):
            with st.container(border=True):
                st.markdown(f"#### #{i} 🍽️ {p.get('menu_name','(메뉴명 없음)')}")
                st.caption(f"식당: {p.get('restaurant','')}")
                st.write("**이유**")
                st.write("- " + "\n- ".join(p.get("why", []) or ["(이유 없음)"]))

                st.write("**추천 근거(규칙 트레이스)**")
                rules = p.get("evidence_rules", []) or []
                if rules:
                    for r in rules:
                        st.markdown(
                            f"- **IF** {r.get('if','')}  \n"
                            f"  **THEN** {r.get('then','')}  \n"
                            f"  **BECAUSE** {r.get('because','')}"
                        )
                else:
                    st.write("(근거 규칙이 비어있음)")

                conf = p.get("confidence", 0.0)
                st.progress(min(max(float(conf), 0.0), 1.0))
                st.caption(f"신뢰도(모델 추정): {conf}")

                cautions = p.get("cautions", []) or []
                if cautions:
                    st.warning("주의사항: " + " / ".join(cautions))

                # 피드백 버튼
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button(f"👍 좋았어요 (#{i})", key=f"like_{i}"):
                        save_feedback(conn, {
                            "created_at": dt.datetime.utcnow().isoformat(),
                            "user_id": user_id,
                            "country": country,
                            "city": city,
                            "restaurant": p.get("restaurant",""),
                            "recommended_menu": p.get("menu_name",""),
                            "chosen_menu": p.get("menu_name",""),
                            "sentiment": "like",
                            "notes": "",
                            "context_json": {"weather": weather_used, "taste": taste, "condition": condition, "menu_source": menu_source},
                        })
                        st.success("피드백 저장 완료! (좋았어요)")
                with c2:
                    if st.button(f"👎 별로였어요 (#{i})", key=f"dislike_{i}"):
                        save_feedback(conn, {
                            "created_at": dt.datetime.utcnow().isoformat(),
                            "user_id": user_id,
                            "country": country,
                            "city": city,
                            "restaurant": p.get("restaurant",""),
                            "recommended_menu": p.get("menu_name",""),
                            "chosen_menu": p.get("menu_name",""),
                            "sentiment": "dislike",
                            "notes": "",
                            "context_json": {"weather": weather_used, "taste": taste, "condition": condition, "menu_source": menu_source},
                        })
                        st.success("피드백 저장 완료! (별로였어요)")
                with c3:
                    chosen = st.text_input(f"내가 실제로 고른 메뉴 (#{i})", value="", key=f"chosen_{i}", placeholder="예: Pho Bo / Pad Thai 등")
                    if st.button(f"✅ 선택 저장 (#{i})", key=f"save_choice_{i}") and chosen.strip():
                        save_feedback(conn, {
                            "created_at": dt.datetime.utcnow().isoformat(),
                            "user_id": user_id,
                            "country": country,
                            "city": city,
                            "restaurant": p.get("restaurant",""),
                            "recommended_menu": p.get("menu_name",""),
                            "chosen_menu": chosen.strip(),
                            "sentiment": "neutral",
                            "notes": "",
                            "context_json": {"weather": weather_used, "taste": taste, "condition": condition, "menu_source": menu_source},
                        })
                        st.success("선택 메뉴 저장 완료!")

        st.markdown("### 🧠 왜 이 추천을 신뢰해도 되나요?")
        st.write("**사용한 데이터:** " + ", ".join(trust.get("data_used", []) or []))
        lim = trust.get("limitations", []) or []
        if lim:
            st.info("**한계/불확실성:** " + " / ".join(lim))

        if follow:
            st.markdown("### ❓ 정확도 높이기 질문")
            for q in follow:
                st.write("- " + q)

# 메뉴 후보 표시(디버깅/투명성)
with st.expander("🔎 사용된 메뉴 후보 보기(투명성)", expanded=False):
    st.write(f"메뉴 소스: {menu_source}")
    if menu_items:
        st.dataframe(pd.DataFrame(menu_items[:100]), use_container_width=True)
    else:
        st.write("메뉴 후보가 없습니다. 메뉴 DB를 업로드하거나 메뉴판 사진을 올려주세요.")
