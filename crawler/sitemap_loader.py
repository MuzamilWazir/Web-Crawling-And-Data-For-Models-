# crawler/sitemap_loader.py
import requests
import xml.etree.ElementTree as ET

def load_sitemap(sitemap_url: str) -> list[str]:
    response = requests.get(sitemap_url, timeout=20)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls = []
    for url in root.findall("ns:url", ns):
        loc = url.find("ns:loc", ns)
        if loc is not None:
            urls.append(loc.text)

    return urls
