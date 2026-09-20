import requests
import xml.etree.ElementTree as ET


RSS_URL = "	https://www.pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3&reg=48"


def fetch_rss():
    response = requests.get(
        RSS_URL,
        timeout=30,
        headers={
            "User-Agent": "IndianPolityAI/1.0"
        }
    )

    response.raise_for_status()

    return response.content


def parse_rss(xml_data):
    root = ET.fromstring(xml_data)

    articles = []

    for item in root.findall(".//item"):
        article = {
            "title": item.findtext("title"),
            "link": item.findtext("link"),
            "description": item.findtext("description"),
            "pub_date": item.findtext("pubDate"),
        }

        articles.append(article)

    return articles


if __name__ == "__main__":
    xml_data = fetch_rss()

    articles = parse_rss(xml_data)

    print(f"Found {len(articles)} articles\n")

    for article in articles[:10]:
        print("=" * 80)
        print("TITLE:", article["title"])
        print("LINK:", article["link"])
