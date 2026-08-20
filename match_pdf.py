import csv
import glob
import os
import re
import pdfplumber

# フォルダー内のPDFファイルを自動検索
pdf_files = glob.glob("*.pdf")
web_csv_path = "formatted_result.csv"
output_csv_path = "final_result.csv"

pdf_data = {}

if pdf_files:
    pdf_path = pdf_files[0]
    print(f"PDFファイル '{pdf_path}' を解析中...")

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()

                # 行頭の出品番号（例: 15-10）を取得
                no_match = re.search(r"^(\d+-\d+)", line)
                if no_match:
                    item_no = no_match.group(1)

                    # 行内から金額（最後の¥表記）を取得
                    prices = re.findall(r"¥[\d,]+", line)
                    if prices:
                        pdf_data[item_no] = prices[-1]  # 落札金額（¥...）のみを登録
                    elif "未落札" in line:
                        pdf_data[item_no] = "未落札"
                    elif "保留" in line:
                        pdf_data[item_no] = "保留"
                    elif "取消" in line or "キャンセル" in line:
                        pdf_data[item_no] = "出品取消"
                    elif "流れ" in line:
                        pdf_data[item_no] = "流れ"

    print(f"PDFから {len(pdf_data)} 件のデータを抽出しました。")
else:
    print("※ PDFファイルが見つかりません。")

# 2. Web抽出データと結合（落札データ＝「¥」から始まるもののみ抽出）
final_rows = []
skip_count = 0

if os.path.exists(web_csv_path):
    with open(web_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_no = row.get("出品番号", "")
            price = pdf_data.get(item_no, "対象外/未掲載")

            # --- 修正箇所: 落札金額（¥始まり）以外は除外する ---
            if price.startswith("¥"):
                final_rows.append(
                    {
                        "出品番号": item_no,
                        "落札金額": price,
                        "画像URL": row.get("画像URL", ""),
                        "元情報": row.get("元情報", ""),
                        "ページ": row.get("ページ", ""),
                    }
                )
            else:
                skip_count += 1

    # 3. final_result.csv へ出力
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["出品番号", "落札金額", "画像URL", "元情報", "ページ"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(
        f"処理完了: 「未落札」など {skip_count} 件を除外しました。"
    )
    print(
        f"実際に落札された {len(final_rows)} 件のデータを '{output_csv_path}' に保存しました。"
    )