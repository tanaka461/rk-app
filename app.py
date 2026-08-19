import streamlit as st
import sqlite3
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import base64
import streamlit.components.v1 as components

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
    sub_tab1, sub_tab2 = st.tabs(["📸 カメラで撮影して検索", "📁 画像をアップロード"])
    
    with sub_tab1:
        st.write("背面カメラで商品を撮影してください")
        
        camera_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            .cam-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
                background-color: #111;
                border-radius: 16px;
                padding: 12px;
                box-sizing: border-box;
            }
            video {
                width: 100%;
                height: 450px;
                object-fit: cover;
                border-radius: 12px;
                background-color: #000;
            }
            .controls {
                margin-top: 12px;
                margin-bottom: 5px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .shutter-btn {
                width: 72px;
                height: 72px;
                background-color: #ff3b30;
                border: 5px solid #ffffff;
                border-radius: 50%;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                cursor: pointer;
                outline: none;
                transition: transform 0.1s;
            }
            .shutter-btn:active {
                transform: scale(0.92);
                background-color: #d70015;
            }
            .btn-label {
                color: #ccc;
                font-size: 13px;
                font-weight: bold;
                margin-top: 6px;
            }
        </style>
        </head>
        <body>
        <div class="cam-container">
            <video id="webcam" autoplay playsinline></video>
            <canvas id="canvas" style="display:none;"></canvas>
            <div class="controls">
                <button class="shutter-btn" id="snap-btn" onclick="takePhoto()"></button>
                <div class="btn-label">タップして撮影</div>
            </div>
        </div>

        <script>
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');

            navigator.mediaDevices.getUserMedia({
                video: { facingMode: { exact: "environment" } },
                audio: false
            }).catch(function(err) {
                return navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "environment" },
                    audio: false
                });
            }).catch(function(err) {
                return navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });
            }).then(function(stream) {
                video.srcObject = stream;
            }).catch(function(err) {
                console.error("Camera access error:", err);
            });

            function takePhoto() {
                const targetWidth = 600;
                const aspect = (video.videoHeight || 480) / (video.videoWidth || 640);
                
                canvas.width = targetWidth;
                canvas.height = targetWidth * aspect;
                
                const context = canvas.getContext('2d');
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
                
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: dataUrl
                }, '*');
            }
        </script>
        </body>
        </html>
        """
        camera_data = components.html(camera_html, height=580)
        
        # 判定を修正：シャッターを押して有効な文字列が入ってきた時だけデコードを実行する
        if camera_data and isinstance(camera_data, str) and len(camera_data) > 100:
            try:
                raw_str = camera_data
                if "base64," in raw_str:
                    base64_str = raw_str.split("base64,")[1]
                elif "," in raw_str:
                    base64_str = raw_str.split(",")[1]
                else:
                    base64_str = raw_str
                
                base64_str = base64_str.strip()
                missing_padding = len(base64_str) % 4
                if missing_padding:
                    base64_str += '=' * (4 - missing_padding)

                img_data = base64.b64decode(base64_str)
                uploaded_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            except Exception:
                st.error("画像の処理中にエラーが発生しました。もう一度撮影してください。")

    with sub_tab2:
        file_input = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png", "webp"])
        if file_input:
            uploaded_image = Image.open(file_input).convert("RGB")

# --- 4. 検索結果表示 ---
if uploaded_image:
    st.divider()
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(uploaded_image, caption="検索画像", use_container_width=True)
    with col_info:
        st.info("AIモデルを準備して画像を解析中...")

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

elif search_keyword.strip():
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