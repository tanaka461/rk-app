import csv
import re

input_csv = "result.csv"
output_csv = "formatted_result.csv"

formatted_data = []

with open(input_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        info = row.get("info", "")
        img_url = row.get("img_url", "")
        page = row.get("page", "")

        # 先頭の数字パターン（例: "1 1" や "1-1"）を "1-1" 形式に正規化
        match = re.search(r"^(\d+)[\s\-](\d+)", info.strip())
        if match:
            item_no = f"{match.group(1)}-{match.group(2)}"
        else:
            item_no = "不明"

        formatted_data.append(
            {
                "出品番号": item_no,
                "元情報": info,
                "画像URL": img_url,
                "ページ": page,
            }
        )

# 整形後のデータを保存
with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    fieldnames = ["出品番号", "元情報", "画像URL", "ページ"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(formatted_data)

print(
    f"整形完了: {len(formatted_data)} 件のデータを '{output_csv}' に保存しました。"
)