from bs4 import BeautifulSoup

with open("page_source.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

# 画像タグとその周りの構造を上位5件表示
imgs = soup.find_all("img")
print(f"--- 検出された画像総数: {len(imgs)} ---")
for i, img in enumerate(imgs[:5]):
    print(f"[画像 {i+1}]")
    print("  親タグ:", img.parent.name, "class:", img.parent.get("class"))
    print("  src:", img.get("src"))

# ページ送りリンクの候補を表示
print("\n--- ページ送り（次へ）の候補 ---")
for a in soup.find_all("a"):
    text = a.text.strip()
    href = a.get("href", "")
    cls = a.get("class", "")
    if any(k in text or k in str(cls) or k in href for k in ["次", ">", "next", "page"]):
        print(f"  テキスト: '{text}' | class: {cls} | href: {href}")