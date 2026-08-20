import csv
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 1. ブラウザの起動設定
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

csv_file = "result.csv"

try:
    base_url = "https://member.gt-auc.jp/auction/nyuusatsu/list/940?max_disp=0&auction_schedule_id=940&page=1"
    driver.get(base_url)
    
    print("--------------------------------------------------")
    print("ブラウザでログインを完了させてください。")
    print("ログイン後、出品一覧が表示されたらここで【Enterキー】を押してください。")
    print("--------------------------------------------------")
    input()

    # CSVヘッダーの初期作成（上書きリセット）
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "info", "img_url"])
        writer.writeheader()

    total_pages = 271
    total_count = 0

    # 2. 全271ページを巡回
    for page in range(1, total_pages + 1):
        url = f"https://member.gt-auc.jp/auction/nyuusatsu/list/940?max_disp=0&auction_schedule_id=940&page={page}"
        driver.get(url)
        time.sleep(2)  # 描画待ち

        soup = BeautifulSoup(driver.page_source, "html.parser")
        items = soup.find_all("a", class_="underline")
        
        page_results = []
        for item in items:
            img_tag = item.find("img")
            if not img_tag:
                continue

            # 画像URLの取得
            img_url = img_tag.get("src") or img_tag.get("data-src", "")
            if not img_url or "dummy" in img_url or "logo" in img_url:
                continue

            # テキスト情報の取得（画像の親要素・カード枠全体から商品番号などを取得）
            card_element = item.find_parent("li") or item.find_parent("tr") or item.parent.parent
            if card_element:
                text_info = card_element.get_text(separator=" ", strip=True)
            else:
                text_info = item.get_text(strip=True)

            page_results.append({
                "page": page,
                "info": text_info,
                "img_url": img_url
            })

        # 1ページごとにCSVへ即時追記（途中で止めても保存されます）
        if page_results:
            with open(csv_file, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["page", "info", "img_url"])
                writer.writerows(page_results)

        total_count += len(page_results)
        print(f"[{page}/{total_pages} ページ目] {len(page_results)} 件取得 (累計: {total_count} 件)")

    print(f"\n処理完了: 合計 {total_count} 件のデータを '{csv_file}' に保存しました。")

finally:
    driver.quit()