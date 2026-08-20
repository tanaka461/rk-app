import csv
import re
import sqlite3
import urllib.request
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# 1. AIモデルのロード (Streamlitアプリと同じモデル)
print("AIモデル (CLIP) を読み込んでいます...")
MODEL_NAME = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def get_image_vector_from_url(url):
    """画像URLから画像をダウンロードし、ベクトルを計算する"""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            image_bytes = resp.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

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

        return feats.numpy().flatten().astype(np.float32).tobytes()
    except Exception as e:
        return None


# 2. データベースの初期化・テーブル作成
db_file = "rk_data.db"
conn = sqlite3.connect(db_file)
c = conn.cursor()

c.execute(
    """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    model TEXT,
    price INTEGER,
    img_url TEXT,
    vector BLOB,
    date TEXT
)
"""
)
conn.commit()

# 3. CSVの読み込みと登録処理
csv_file = "final_result.csv"
target_market_date = "GT ｜ 2026-08-15"  # 市場名と日付をセットで保存

print(f"'{csv_file}' からデータを読み込み、'{db_file}' へ登録を開始します...\n")

success_count = 0
vector_count = 0

with open(csv_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    total = len(rows)

    for idx, row in enumerate(rows, 1):
        item_no = row.get("出品番号", "")
        title = row.get("元情報", "")
        img_url = row.get("画像URL", "")
        price_str = row.get("落札金額", "")

        # 金額数値化 ("¥930,000" -> 930000)
        clean_price = re.sub(r"[^\d]", "", price_str)
        price_val = int(clean_price) if clean_price else 0

        # 型番抽出 (タイトル内の「型番： XXX」などがあれば拾う、無ければ出品番号)
        model_match = re.search(r"型番：\s*([^\s]+)", title)
        if model_match and model_match.group(1) != "-":
            model_val = model_match.group(1)
        else:
            model_val = item_no

        # 画像から特徴量ベクトルを取得
        vector_bytes = get_image_vector_from_url(img_url)
        if vector_bytes:
            vector_count += 1

        # DBにインサート（date列に "GT ｜ 2026-08-15" を指定）
        c.execute(
            """
            INSERT INTO items (title, model, price, img_url, vector, date)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                title,
                model_val,
                price_val,
                img_url,
                vector_bytes,
                target_market_date,
            ),
        )

        success_count += 1

        if idx % 50 == 0 or idx == total:
            conn.commit()
            print(
                f"進捗: [{idx}/{total}] 件完了 (画像ベクトル抽出成功: {vector_count}件)"
            )

conn.commit()
conn.close()

print(f"\n✨ 登録が完了しました！")
print(f"・総登録件数: {success_count} 件")
print(f"・画像検索対応 (ベクトル保存済み): {vector_count} 件")
print(f"・登録区分: {target_market_date}")
print(f"・保存先データベース: {db_file}")