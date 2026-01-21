# crawler/selenium_loader.py
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException


class SeleniumLoader:
    def __init__(self, page_load_timeout=25):
        self.page_load_timeout = page_load_timeout
        self._start_driver()

    def _start_driver(self):
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")

        # 👇 Fake a real browser
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(self.page_load_timeout)

    def load_page_text(self, url: str, retries=2) -> str:
        for attempt in range(1, retries + 1):
            try:
                self.driver.get(url)
                time.sleep(3)  # allow JS to settle

                soup = BeautifulSoup(self.driver.page_source, "html.parser")

                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text = soup.get_text(separator=" ", strip=True)

                if not text:
                    raise RuntimeError("Empty page")

                return text

            except WebDriverException as e:
                if attempt == retries:
                    raise RuntimeError(f"Page load failed: {url}") from e

                # 🔁 Restart driver on failure
                self._restart_driver()

    def _restart_driver(self):
        try:
            self.driver.quit()
        except Exception:
            pass
        time.sleep(2)
        self._start_driver()

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass
