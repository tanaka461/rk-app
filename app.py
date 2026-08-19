import streamlit as st
import sqlite3
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# --- 1. ページ設定 & CSSによる翻訳の完全拒否 ---
st.set_page_config(page_title="相場検索アプリ", layout="wide")

st.markdown("""
    <style>
        html, body, .stApp, div, p, span, h1, h2, h3 {
            translate: no !important;
        }
    </style>
    <meta name="google" content="notranslate">
""", unsafe_allow_html=True)

st.title("👜 相場検索アプリ")

# --- 2. AIモデル (遅延ロード対応) & DB設定 ---
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

# ベクトル（画像）類似度検索
def search_similar(query_vector, top_k=12):
    conn = sqlite3.connect("rk_data.db")
    c = conn.cursor()
    c.execute("SELECT id, title, model, price, img_url, vector, date FROM items")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return []

    results = []
    for row in rows:
        item_id, title, model_num, price, img_url, vec_bytes, date_str = row
        if not vec_bytes:
            continue
        db_vec = np.frombuffer(vec_bytes, dtype=np.float32)
        sim = np.dot(query_vector, db_vec) / (np.linalg.norm(query_vector) * np.linalg.norm(db_vec))
        results.append({
            "title": title,
            "model": model_num,
            "price": price,
            "img_url": img_url,
            "date": date_str,
            "similarity": float(sim)
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# テキスト（型番・キーワード）検索（AND検索 & 表示50件）
def search_by_keyword(keyword, top_k=50):
    conn = sqlite3.connect("rk_data.db")
    c = conn.cursor()
    
    keywords = keyword.replace(" ", " ").split()
    if not keywords:
        conn.close()
        return []

    where_clauses = []
    params = []
    for kw in keywords:
        where_clauses.append("(title LIKE ? OR model LIKE ?)")
        search_pattern = f"%{kw}%"
        params.extend([search_pattern, search_pattern])
        
    where_sql = " AND ".join(where_clauses)
    query = f"""
        SELECT title, model, price, img_url, date 
        FROM items 
        WHERE {where_sql}
        ORDER BY id DESC
        LIMIT ?
    """
    params.append(top_k)
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    results = []
    for row in rows:
        title, model_num, price, img_url, date_str = row
        results.append({
            "title": title,
            "model": model_num,
            "price": price,
            "img_url": img_url,
            "date": date_str
        })
    return results

# --- 3. タブUI構成 ---
tab_main1, tab_main2 = st.tabs(["🔍 型番・キーワード検索", "📷 画像で検索"])

uploaded_image = None
search_keyword = None

with tab_main1:
    search_keyword = st.text_input("商品名や型番を入力 (例: バーキン トゴ, M43735)", "", key="kw_input")

with tab_main2:
    sub_tab1, sub_tab2 = st.tabs(["📸 アプリ内で撮影して検索", "📁 画像をアップロード"])
    
    with sub_tab1:
        st.write("画面のカメラで商品を撮影してください")
        camera_file = st.camera_input("カメラ撮影", key="native_camera_input")
        if camera_file:
            uploaded_image = Image.open(camera_file).convert("RGB")

    with sub_tab2:
        file_input = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png", "webp"])
        if file_input:
            uploaded_image = Image.open(file_input).convert("RGB")

# --- 4. 検索結果表示 ---
if uploaded_image:
    st.divider()
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(uploaded_image, caption="撮影／選択画像", use_container_width=True)
    with col_info:
        st.info("AIモデルで特徴量を抽出して検索中...")

    query_vec = get_image_features(uploaded_image)
    results = search_similar(query_vec, top_k=12)

    st.subheader("🎉 AI類似検索結果（上位12件）")
    if not results:
        st.warning("該当するデータが見つかりませんでした。")
    else:
        cols = st.columns(3)
        for idx, item in enumerate(results):
            with cols[idx % 3]:
                st.image(item["img_url"], use_container_width=True)
                st.markdown(f"**{item['title']}**")
                if item["model"]:
                    st.caption(f"型番: {item['model']}")
                st.subheader(f"¥{item['price']:,}")
                st.caption(f"🏛️ **RK** ｜ 📅 {item['date']}")
                st.caption(f"類似度: {item['similarity']*100:.1f}%")
                st.divider()

elif search_keyword and search_keyword.strip():
    st.divider()
    kw_results = search_by_keyword(search_keyword.strip(), top_k=50)
    st.subheader(f"🔍 「{search_keyword}」の検索結果 ({len(kw_results)} 件)")

    if not kw_results:
        st.warning("一致する商品が見つかりませんでした。別のキーワードでお試しください。")
    else:
        cols = st.columns(3)
        for idx, item in enumerate(kw_results):
            with cols[idx % 3]:
                st.image(item["img_url"], use_container_width=True)
                st.markdown(f"**{item['title']}**")
                if item["model"]:
                    st.caption(f"型番: {item['model']}")
                st.subheader(f"¥{item['price']:,}")
                st.caption(f"🏛️ **RK** ｜ 📅 {item['date']}")
                st.divider()