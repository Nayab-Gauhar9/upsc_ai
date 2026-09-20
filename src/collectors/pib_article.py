import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from src.collectors.pib_rss import fetch_rss, parse_rss
from src.processors.normalize_validate import normalize_record, validate_record


BASE_URL = "https://www.pib.gov.in/PressReleasePage.aspx"


def build_url(prid):
    return f"{BASE_URL}?PRID={prid}&reg=48&lang=1"


def fetch_article(url):
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    return response.text


def detect_language(soup):
    text = soup.get_text(" ", strip=True)

    hindi_chars = 0
    english_chars = 0

    for char in text:
        if "\u0900" <= char <= "\u097F":
            hindi_chars += 1

        elif ("A" <= char <= "Z") or ("a" <= char <= "z"):
            english_chars += 1

    if hindi_chars > english_chars:
        return "hi"

    return "en"


def find_english_url(soup):
    for link in soup.find_all("a", href=True):

        text = link.get_text(" ", strip=True).lower()

        if text == "english":

            return link["href"]

    return None


def extract_prid(url):
    parsed_url = urlparse(url)

    query = parse_qs(parsed_url.query)

    prid = query.get("PRID")

    if prid:
        return prid[0]

    return None


def resolve_article(prid):

    original_url = build_url(prid)

    print("=" * 80)
    print("ORIGINAL PRID:", prid)
    print("ORIGINAL URL:", original_url)

    html = fetch_article(original_url)

    soup = BeautifulSoup(html, "html.parser")

    language = detect_language(soup)

    print("DETECTED LANGUAGE:", language)

    # Article is already English
    if language == "en":

        print("ARTICLE IS ALREADY ENGLISH")

        return {
            "source_prid": prid,
            "language": "en",
            "english_prid": prid,
            "english_url": original_url,
            "soup": soup
        }

    # Article is not English
    print("ARTICLE IS NOT ENGLISH")

    english_url = find_english_url(soup)

    if not english_url:

        print("NO ENGLISH VERSION FOUND")

        return {
            "source_prid": prid,
            "language": language,
            "english_prid": None,
            "english_url": None,
            "soup": None
        }

    english_prid = extract_prid(english_url)

    # Fetch English version
    english_html = fetch_article(english_url)

    english_soup = BeautifulSoup(
        english_html,
        "html.parser"
    )

    english_language = detect_language(english_soup)

    print("ENGLISH PAGE LANGUAGE:", english_language)
    print("ENGLISH URL:", english_url)
    print("ENGLISH PRID:", english_prid)

    return {
        "source_prid": prid,
        "language": language,
        "english_prid": english_prid,
        "english_url": english_url,
        "soup": english_soup
    }


def extract_basic_metadata(soup):

    title = None
    published_text = None

    # Article title
    heading = soup.find("h2")

    if heading:
        title = heading.get_text(" ", strip=True)

    # Publication date
    date_element = soup.find(id ="PrDateTime")
    if date_element:
        published_text= " ".join(date_element.get_text(" ", strip=True).split())

    return {
        "title": title,
        "published_at": published_text
    }

def extract_article_content(soup):

    date_element = soup.find(id="PrDateTime")

    if not date_element:
        print("DATE ELEMENT NOT FOUND")
        return ""

    article_container = date_element.find_parent(
        "div",
        class_="innner-page-main-about-us-content-right-part"
    )

    if not article_container:
        print("ARTICLE CONTAINER NOT FOUND")
        return ""

    children = article_container.find_all(recursive=False)

    article_parts = []
    collecting = False

    for child in children:

        if child == date_element:
            collecting = True
            continue

        if not collecting:
            continue

        text = child.get_text(" ", strip=True)

        if text:
            article_parts.append(text)

    # Join everything first
    article_text = "\n\n".join(article_parts)

    # -----------------------------------------
    # Remove everything from *** onwards
    # -----------------------------------------

    marker_position = article_text.find("***")

    if marker_position != -1:
        article_text = article_text[:marker_position].rstrip()

    return article_text

def collect_article(prid):

    result = resolve_article(prid)

    if result["soup"] is None:
        return None

    metadata = extract_basic_metadata(
        result["soup"]
    )

    article = extract_article_content(
        result["soup"]
    )

    return {
        "source": "PIB",
        "source_prid": result["source_prid"],
        "source_language": result["language"],
        "english_prid": result["english_prid"],
        "url": result["english_url"],
        "title": metadata["title"],
        "published_at": metadata["published_at"],
        "article_text": article
    }

if __name__ == "__main__":

    xml_data = fetch_rss()
    articles = parse_rss(xml_data)

    records = []

    print("=" * 80)
    print(f"RSS ARTICLES FOUND: {len(articles)}")
    print("=" * 80)

    for index, item in enumerate(articles, start=1):

        rss_url = item["link"]
        prid = extract_prid(rss_url)

        print(f"\n[{index}/{len(articles)}] PRID: {prid}")

        if not prid:
            print("PRID NOT FOUND")
            continue

        try:

            record = collect_article(prid)

            if record is None:
                print("NO ENGLISH VERSION")
                continue
            normalized = normalize_record(record)
            valid,error = validate_record(normalized)
            if not valid:
                print("VALIDATION FAILED:", error)
                continue
            records.append(normalized)

            print("STATUS: VALID")
            print("TITLE:", record["title"])

        except Exception as e:

            print("ERROR:", e)


    print("\n" + "=" * 100)
    print("NORMALIZED RECORD VALIDATION")
    print("=" * 100)

    print("TOTAL RECORDS:", len(records))

    for index, record in enumerate(records, start=1):

        print("\n" + "-" * 100)
        print(f"RECORD {index}")

        print("SOURCE:", record["source"])
        print("SOURCE PRID:", record["source_prid"])
        print("SOURCE LANGUAGE:", record["source_language"])
        print("ENGLISH PRID:", record["english_prid"])
        print("TITLE:", record["title"])
        print("PUBLISHED:", record["published_at"])

        print(
            "ARTICLE LENGTH:",
            len(record["article_text"])
        )
