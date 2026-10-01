import json
import requests
from bs4 import BeautifulSoup
from src.processors.normalize_validate import normalize_record, validate_record

PRID = "2317557"
URL = f"https://www.pib.gov.in/PressReleasePage.aspx?PRID={PRID}&reg=48&lang=1"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

print("=" * 80)
print(f"FETCHING RAW HTML FOR PRID: {PRID}")
print("=" * 80)

resp = requests.get(URL, headers=headers, timeout=30)
print(f"HTTP Status: {resp.status_code}")

soup = BeautifulSoup(resp.text, "html.parser")

# 1. Inspect Title Elements
print("\n--- 1. TITLE / HEADER TAGS FOUND IN HTML ---")
h2_tags = soup.find_all("h2")
print(f"All <h2> tags count: {len(h2_tags)}")
for i, h in enumerate(h2_tags):
    print(f"  h2[{i}]: text='{h.get_text(strip=True)}' | class={h.get('class')}")

h3_tags = soup.find_all("h3")
print(f"All <h3> tags count: {len(h3_tags)}")
for i, h in enumerate(h3_tags):
    print(f"  h3[{i}]: text='{h.get_text(strip=True)}' | class={h.get('class')}")

doc_title = soup.title.get_text(strip=True) if soup.title else None
print(f"Page <title> tag: '{doc_title}'")

# 2. Inspect Date Element
print("\n--- 2. DATE ELEMENT (PrDateTime) ---")
date_el = soup.find(id="PrDateTime")
if date_el:
    print(f"PrDateTime raw text: '{date_el.get_text(strip=True)}'")
else:
    print("PrDateTime element NOT FOUND")

# 3. Inspect Body Content
print("\n--- 3. ARTICLE CONTENT CONTAINER ---")
container = None
if date_el:
    container = date_el.find_parent("div", class_="innner-page-main-about-us-content-right-part")

if container:
    children = container.find_all(recursive=False)
    print(f"Found container with {len(children)} top-level child blocks.")
    
    parts = []
    collecting = False
    for child in children:
        if child == date_el:
            collecting = True
            continue
        if not collecting:
            continue
        text = child.get_text(" ", strip=True)
        if text:
            parts.append(text)
            
    full_text = "\n\n".join(parts)
    marker_pos = full_text.find("***")
    if marker_pos != -1:
        full_text = full_text[:marker_pos].rstrip()

    print(f"Extracted Character Length: {len(full_text)}")
    print("\n--- FIRST 350 CHARACTERS OF BODY TEXT ---")
    print(full_text[:350] + "...")
else:
    print("Target container 'innner-page-main-about-us-content-right-part' NOT FOUND")
