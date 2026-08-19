import sqlite3

print("データベースの重複データを整理しています...")

conn = sqlite3.connect("rk_data.db")
c = conn.cursor()

# 画像URLまたは (タイトル・価格) が重複している場合に1件だけ残して削除
c.execute('''
    DELETE FROM items
    WHERE id NOT IN (
        SELECT MIN(id)
        FROM items
        GROUP BY img_url, title, price
    )
''')

deleted_count = c.rowcount
conn.commit()

# DBファイルの容量を自動圧縮
c.execute("VACUUM")
conn.close()

print(f"🎉 完了！ 重複していた {deleted_count} 件のデータを削除してスッキリさせました！")