import csv

with open("final_result.csv", encoding="utf-8-sig") as f:
    reader = list(csv.DictReader(f))

# 「対象外/未掲載」になっているデータを抽出
unmatched = [row for row in reader if row["落札金額"] == "対象外/未掲載"]

print(f"▼ 未マッチの合計件数: {len(unmatched)}件\n")
print("--- 先頭10件の詳細 ---")
for x in unmatched[:10]:
    print(f"ページ:{x['ページ']} | 出品番号:{x['出品番号']} | 情報:{x['元情報'][:50]}")