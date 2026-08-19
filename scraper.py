import sqlite3
import time
import re
import requests
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from playwright.sync_api import sync_playwright

# --- 1. AIモデルの準備（高精度版 patch16） ---
MODEL_NAME = "openai/clip-vit-base-patch16"
print(f"AIモデル [{MODEL_NAME}] を準備しています...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def get_image_vector(image_url):
    """画像URLから画像をダウンロードし、AI特徴量（ベクトル）に変換"""
    try:
        if not image_url.startswith("http"):
            image_url = "https://bid.rk-auction.jp" + image_url

        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        
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
        return None

def save_to_db(title, model_number, price, img_url, date_str):
    """データベース(rk_data.db)へ1件保存（上書き対応）"""
    conn = sqlite3.connect("rk_data.db")
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS items 
                 (id INTEGER PRIMARY KEY, title TEXT, model TEXT, price INT, img_url TEXT, vector BLOB, date TEXT)''')
    
    # 重複・上書きチェック
    c.execute("SELECT id FROM items WHERE img_url = ?", (img_url,))
    row = c.fetchone()

    vector_blob = get_image_vector(img_url)
    if not vector_blob:
        conn.close()
        return False

    if row:
        c.execute("UPDATE items SET title = ?, model = ?, price = ?, vector = ?, date = ? WHERE id = ?",
                  (title, model_number, price, vector_blob, date_str, row[0]))
    else:
        c.execute("INSERT INTO items (title, model, price, img_url, vector, date) VALUES (?, ?, ?, ?, ?, ?)",
                  (title, model_number, price, img_url, vector_blob, date_str))

    conn.commit()
    conn.close()
    print(f"  [+] 登録完了 ({date_str}): {title} | 型番:{model_number} | ¥{price:,}")
    return True

# --- 2. スクレイピング処理 ---
def run_scraper(login_id, password, base_url, target_date):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=200)
        page = browser.new_page()

        print("\n1. ログイン画面を開いています...")
        page.goto("https://bid.rk-auction.jp/")
        page.wait_for_load_state("domcontentloaded")

        print("2. ログイン情報を入力中...")
        try:
            id_selectors = ["input[type='text']", "input[name*='id']", "input[name*='user']", "input[name*='login']", "input"]
            pass_selectors = ["input[type='password']", "input[name*='pass']"]

            for sel in id_selectors:
                if page.is_visible(sel):
                    page.fill(sel, login_id)
                    break

            for sel in pass_selectors:
                if page.is_visible(sel):
                    page.fill(sel, password)
                    break

            submit_btn = page.query_selector("button[type='submit'], input[type='submit'], .login-btn, #login-btn")
            if submit_btn:
                submit_btn.click()
            else:
                page.keyboard.press("Enter")

            time.sleep(3)
        except Exception:
            pass

        total_saved = 0
        page_num = 1
        ignore_words = ["フリーワード", "箱番", "画像", "商品名", "検索結果", "ようこそ", "お知らせ", "お問い合わせ"]

        while True:
            target_url = f"{base_url}&page={page_num}" if "page=" not in base_url else base_url
            print(f"\n==================== ページ {page_num} の処理開始 (設定日付: {target_date}) ====================")
            
            if page_num > 1:
                page.goto(target_url)
            else:
                page.goto(base_url)

            time.sleep(4)
            page.wait_for_load_state("networkidle")

            rows = page.query_selector_all("tr")
            if not rows or len(rows) < 2:
                rows = page.query_selector_all("div")

            page_count = 0
            for row in rows:
                try:
                    text = row.inner_text().strip()
                    if "￥" not in text and "¥" not in text:
                        continue

                    if any(word in text for word in ignore_words):
                        continue

                    price_match = re.search(r'[￥¥]\s*([\d,]+)', text)
                    if not price_match:
                        continue
                    price = int(price_match.group(1).replace(",", ""))

                    img_elem = row.query_selector("img")
                    if not img_elem:
                        continue
                    img_url = img_elem.get_attribute("src") or img_elem.get_attribute("data-src")
                    if not img_url or "logo" in img_url or "icon" in img_url:
                        continue

                    lines = [l.strip() for l in text.split("\n") if l.strip()]
                    clean_lines = [l for l in lines if not re.search(r'[￥¥]|落札|大会|第\d+回|\d+-\d+', l)]

                    if not clean_lines:
                        continue

                    title = " ".join(clean_lines[:2])
                    if any(word in title for word in ignore_words):
                        continue

                    model_match = re.search(r'\b([A-Z]\d{5}|[A-Z0-9]{5,8})\b', text)
                    model_number = model_match.group(1) if model_match else ""

                    if save_to_db(title, model_number, price, img_url, target_date):
                        total_saved += 1
                        page_count += 1

                except Exception:
                    continue

            print(f"ページ {page_num} 完了: 追加・更新 {page_count} 件 (累計: {total_saved} 件)")

            next_btn = page.query_selector("a:has-text('次'), a:has-text('>'), .next a, li.next a, a[rel='next']")
            if next_btn and next_btn.is_visible():
                try:
                    next_btn.click()
                    time.sleep(3)
                    page_num += 1
                except Exception:
                    page_num += 1
            else:
                if page_count == 0 and page_num > 5:
                    print("\n全ページの取得が完了しました！")
                    break
                page_num += 1

        print(f"\n【追加完了】 新たに {total_saved} 件のデータを処理しました。")
        browser.close()

if __name__ == "__main__":
    USER_ID = "003170"
    USER_PASS = "59155915"
    
    # 第558回URL ＆ 日付: 2026/05/04
    TARGET_URL = "https://bid.rk-auction.jp/result/bag/?ei=558&bd=&ne=&ne_t=0&mr=&mr_t=0&bs=&be=&sf=1"
    TARGET_DATE = "2026/05/04"
    
    run_scraper(USER_ID, USER_PASS, TARGET_URL, TARGET_DATE)