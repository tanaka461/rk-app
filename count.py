import sqlite3

conn = sqlite3.connect("rk_data.db")
c = conn.cursor()

c.execute("SELECT COUNT(*) FROM items")
total_count = c.fetchone()[0]

c.execute("SELECT date, COUNT(*) FROM items GROUP BY date")
date_counts = c.fetchall()

print("=" * 35)
print(f"【合計登録件数】 {total_count:,} 件")
print("=" * 35)
print("【日付ごとの内訳】")
for date_str, count in date_counts:
    print(f"  ・{date_str}: {count:,} 件")
print("=" * 35)

conn.close()