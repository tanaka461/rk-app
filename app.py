import sqlite3
import unicodedata
from io import BytesIO
import numpy as np
from PIL import Image
import streamlit as st
import torch
from transformers import CLIPModel, CLIPProcessor

# --- 1. ページ設定 & CSSによる翻訳の完全拒否 ---
st.set_page_config(page_title="相場検索アプリ", layout="wide")

st.markdown(
    """
    <style>
        html, body, .stApp, div, p, span, h1, h2, h3 {
            translate: no !important;
        }
    </style>
    <meta name="google" content="notranslate">
""",
    unsafe_allow_html=True,
)

# --- 2. ブラウザのカメラ起動処理を外カメラに固定 ---
st.components.v1.html(
    """
    <script>
    (function() {
        try {
            const pWin = window.parent;
            if (!pWin || pWin.__rear_camera_patched) return;
            pWin.__rear_camera_patched = true;

            const mediaDevices = pWin.navigator.mediaDevices;
            if (!mediaDevices || !mediaDevices.getUserMedia) return;

            const originalGetUserMedia = mediaDevices.getUserMedia.bind(mediaDevices);

            mediaDevices.getUserMedia = function(constraints) {
                if (constraints && constraints.video) {
                    if (typeof constraints.video === 'boolean') {
                        constraints.video = { facingMode: { ideal: "environment" } };
                    } else if (typeof constraints.video === 'object') {
                        constraints.video.facingMode = { ideal: "environment" };
                    }
                }
                return originalGetUserMedia(constraints);
            };
        } catch (e) {
            console.error("Camera patch error:", e);
        }
    })();
    </script>
""",
    height=0,
)

# --- 3. UIデザイン設定 ---
st.markdown(
    """
    <style>
        [data-testid="stCameraInput"] {
            width: 100% !important;
            max-width: 800px !important;
            margin: 0 auto !important;
            border-radius: 16px !important;
            overflow: hidden !important;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25) !important;
        }
        
        [data-testid="stCameraInput"] video {
            width: 100% !important;
            height: 520px !important;
            object-fit: cover !important;
            border-radius: 12px !important;
        }

        [data-testid="stCameraInput"] button {
            background-color: #ff3b30 !important;
            color: #ffffff !important;
            border: 4px solid #ffffff !important;
            border-radius: 50% !important;
            width: 78px !important;
            height: 78px !important;
            min-height: 78px !important;
            padding: 0 !important;
            margin: 15px auto !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        }

        [data-testid="stCameraInput"] button:active {
            transform: scale(0.90) !important;
            background-color: #d70015 !important;
        }

        [data-testid="stCameraInput"] button * {
            display: none !important;
        }
        [data-testid="stCameraInput"] button::after {
            content: "" !important;
            width: 26px !important;
            height: 26px !important;
            background-color: #ffffff !important;
            border-radius: 50% !important;
            display: block !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("👜 相場検索アプリ")


# --- 表記揺れ対策（半角カナ→全角カナ変換） ---
def normalize_text(text):
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text)).lower()


# --- 価格を数値(int)に変換する補助関数 ---
def clean_price(price_val):
    if price_val is None:
        return 0
    if isinstance(price_val, (int, float)):
        return int(price_val)
    p_str = (
        str(price_val)
        .replace(",", "")
        .replace("¥", "")
        .replace("円", "")
        .replace(" ", "")
        .strip()
    )
    try:
        return int(float(p_str))
    except ValueError:
        return 0


# --- 4. AIモデル & DB設定 ---
@st.cache_resource
def load_model():
    MODEL_NAME = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def get_image_features(image):
    model, processor = load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "image_embeds"):
            feats = outputs.image_embeds
        elif hasattr(outputs, "last_hidden_state"):
            feats = outputs.last_hidden_state[:, 0, :]
        else:
            feats = outputs
    feats = feats / torch.norm(feats, p=2, dim=-1, keepdim=True)
    return feats.numpy().flatten()


def search_similar(query_vector, target_market="すべて"):
    conn = sqlite3.connect("rk_data.db")
    c = conn.cursor()
    c.execute(
        "SELECT id, title, model, price, img_url, vector, date FROM items"
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        return []

    results = []
    for row in rows:
        item_id, title, model_num, price, img_url, vec_bytes, date_str = row
        date_str = str(date_str or "")

        if target_market == "GTのみ" and "GT" not in date_str:
            continue
        elif target_market == "RKのみ" and "GT" in date_str:
            continue

        if not vec_bytes:
            continue

        db_vec = np.frombuffer(vec_bytes, dtype=np.float32)
        sim = np.dot(query_vector, db_vec) / (
            np.linalg.norm(query_vector) * np.linalg.norm(db_vec)
        )
        results.append(
            {
                "title": title,
                "model": model_num,
                "price": clean_price(price),
                "img_url": img_url,
                "date": date_str,
                "similarity": float(sim),
            }
        )

    return results


def search_by_keyword(keyword, target_market="すべて"):
    conn = sqlite3.connect("rk_data.db")
    c = conn.cursor()

    c.execute("SELECT title, model, price, img_url, date FROM items")
    rows = c.fetchall()
    conn.close()

    norm_keyword = normalize_text(keyword)
    keywords = norm_keyword.replace(" ", " ").split()

    results = []
    for row in rows:
        title, model_num, price, img_url, date_str = row
        date_str = str(date_str or "")

        if target_market == "GTのみ" and "GT" not in date_str:
            continue
        elif target_market == "RKのみ" and "GT" in date_str:
            continue

        norm_title = normalize_text(title)
        norm_model = normalize_text(model_num)
        norm_date = normalize_text(date_str)

        match = True
        if keywords:
            for kw in keywords:
                if (
                    (kw not in norm_title)
                    and (kw not in norm_model)
                    and (kw not in norm_date)
                ):
                    match = False
                    break

        if match:
            results.append(
                {
                    "title": title,
                    "model": model_num,
                    "price": clean_price(price),
                    "img_url": img_url,
                    "date": date_str,
                }
            )

    return results


def sort_results(results, sort_option):
    if not results:
        return []

    if sort_option == "📅 日付が新しい順":
        return sorted(
            results, key=lambda x: str(x.get("date", "")), reverse=True
        )
    elif sort_option == "💴 価格が高い順":
        return sorted(results, key=lambda x: int(x.get("price", 0)), reverse=True)
    elif sort_option == "💴 価格が安い順":
        return sorted(results, key=lambda x: int(x.get("price", 0)))
    elif sort_option == "🎯 類似度が高い順":
        return sorted(
            results, key=lambda x: float(x.get("similarity", 0.0)), reverse=True
        )
    return results


# --- 5. タブUI構成 ---
tab_main1, tab_main2 = st.tabs(["🔍 型番・キーワード検索", "📷 画像で検索"])

# --- タブ1: 型番・キーワード検索 ---
with tab_main1:
    col_input, col_market = st.columns([3, 1])
    with col_input:
        search_keyword = st.text_input(
            "商品名や型番を入力 (例: バーキン, ソミュール, M43735)",
            "",
            key="kw_input",
        )
    with col_market:
        market_filter = st.selectbox(
            "市場絞り込み",
            ["すべて", "GTのみ", "RKのみ"],
            key="kw_market",
        )

    if (
        search_keyword and search_keyword.strip()
    ) or market_filter == "GTのみ":
        st.divider()
        kw_results = search_by_keyword(
            search_keyword.strip(), target_market=market_filter
        )

        col_title, col_sort = st.columns([2, 1])
        with col_title:
            st.subheader(f"🔍 検索結果 ({len(kw_results)} 件)")
        with col_sort:
            sort_order = st.selectbox(
                "並び替え",
                [
                    "📅 日付が新しい順",
                    "💴 価格が高い順",
                    "💴 価格が安い順",
                ],
                key="kw_sort",
            )

        # 検索条件に該当する全件を正しくソート
        kw_results = sort_results(kw_results, sort_order)

        if not kw_results:
            st.warning("一致する商品が見つかりませんでした。")
        else:
            cols = st.columns(3)
            # 表示件数を上位200件に絞り込み
            for idx, item in enumerate(kw_results[:200]):
                with cols[idx % 3]:
                    st.image(item["img_url"], use_container_width=True)
                    st.markdown(f"**{item['title']}**")
                    if item["model"]:
                        st.caption(f"型番: {item['model']}")
                    st.subheader(f"¥{item['price']:,}")

                    d_str = item["date"]
                    if "GT" in d_str:
                        st.caption(f"🏛️ **GT** ｜ 📅 2026-08-15")
                    else:
                        st.caption(f"🏛️ **RK** ｜ 📅 {d_str}")

                    st.divider()

# --- タブ2: 画像で検索 ---
with tab_main2:
    col_cam, col_m_img = st.columns([3, 1])
    with col_m_img:
        img_market_filter = st.selectbox(
            "市場絞り込み",
            ["すべて", "GTのみ", "RKのみ"],
            key="img_market",
        )

    sub_tab1, sub_tab2 = st.tabs(
        ["📸 アプリ内で撮影して検索", "📁 画像をアップロード"]
    )
    uploaded_image = None

    with sub_tab1:
        camera_file = st.camera_input("アプリ内カメラ", key="app_camera")
        if camera_file:
            uploaded_image = Image.open(camera_file).convert("RGB")

    with sub_tab2:
        file_input = st.file_uploader(
            "画像ファイルを選択してください",
            type=["jpg", "jpeg", "png", "webp"],
            key="file_up",
        )
        if file_input:
            uploaded_image = Image.open(file_input).convert("RGB")

    if uploaded_image:
        st.divider()
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(
                uploaded_image,
                caption="撮影／選択画像",
                use_container_width=True,
            )
        with col_info:
            st.info("AIモデルで特徴量を抽出して検索中...")

        query_vec = get_image_features(uploaded_image)
        results = search_similar(
            query_vec, target_market=img_market_filter
        )

        col_title, col_sort = st.columns([2, 1])
        with col_title:
            st.subheader("🎉 AI類似検索結果")
        with col_sort:
            img_sort_order = st.selectbox(
                "並び替え",
                [
                    "🎯 類似度が高い順",
                    "📅 日付が新しい順",
                    "💴 価格が高い順",
                    "💴 価格が安い順",
                ],
                key="img_sort",
            )

        # 全結果を並び替えてから上位18件を表示
        sorted_results = sort_results(results, img_sort_order)[:18]

        if not sorted_results:
            st.warning("該当するデータが見つかりませんでした。")
        else:
            cols = st.columns(3)
            for idx, item in enumerate(sorted_results):
                with cols[idx % 3]:
                    st.image(item["img_url"], use_container_width=True)
                    st.markdown(f"**{item['title']}**")
                    if item["model"]:
                        st.caption(f"型番: {item['model']}")
                    st.subheader(f"¥{item['price']:,}")

                    d_str = item["date"]
                    if "GT" in d_str:
                        st.caption(f"🏛️ **GT** ｜ 📅 2026-08-15")
                    else:
                        st.caption(f"🏛️ **RK** ｜ 📅 {d_str}")

                    st.caption(f"類似度: {item['similarity']*100:.1f}%")
                    st.divider()