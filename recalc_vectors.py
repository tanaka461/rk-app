import sqlite3
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import time

# 高精度モデルに切り替え
MODEL_NAME = "openai/clip-vit-base-patch16"

print(f"モデル [{MODEL_NAME}] を読み込んでいます...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("モデルの読み込み完了！")

def get_image_features_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
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
        return feats.numpy().flatten().tobytes()
    except Exception as e:
        print(f"画像取得エラー: {e}")
        return None

# DB処理
conn = sqlite3.connect("rk_data.db")
c = conn.cursor()

c.execute("SELECT id, img_url FROM items")
items = c.fetchall()
total = len(items)

print(f"全 {total} 件のデータのベクトル（画像特徴量）を更新します...")

success_count = 0
start_time = time.time()

for idx, (item_id, img_url) in enumerate(items, 1):
    vector_bytes = get_image_features_from_url(img_url)
    if vector_bytes:
        c.execute("UPDATE items SET vector = ? WHERE id = ?", (vector_bytes, item_id))
        success_count += 1

    # 50件ごとに保存
    if idx % 50 == 0:
        conn.commit()
        elapsed = time.time() - start_time
        print(f"進捗: {idx}/{total} 件完了 (成功: {success_count}件) - 経過時間: {int(elapsed)}秒")

conn.commit()
conn.close()

print(f"\n🎉 すべての再計算が完了しました！(成功: {success_count}/{total}件)")