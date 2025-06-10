import time
from io import StringIO
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException


class SeleniumResponse:
    def __init__(self, text):
        self.text = text


def fetch_data(url):
    """
    Fetch page HTML using Selenium and Chrome WebDriver.
    Retries on failure with a delay.
    """
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("window-size=1920,1080")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        # Wait for JavaScript-driven content to load
        time.sleep(2)
        html = driver.page_source
        driver.quit()
        return SeleniumResponse(html)
    except WebDriverException as e:
        print(f"WebDriver error: {e}. Retrying in 60 seconds...")
        time.sleep(60)
        return fetch_data(url)
    except Exception as e:
        print(f"Unexpected error fetching data: {e}")
        return None


def fetch_table(url, table_id, label):
    """
    Fetch a single table by ID from the given URL and return as a DataFrame.
    """
    data = fetch_data(url)
    if not data:
        return None

    html = data.text if hasattr(data, "text") else data
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"class": "stats_table", "id": table_id})
    if not table:
        print(f"Could not find the table with id '{table_id}'.")
        return None

    df = pd.read_html(StringIO(str(table)))[0]
    # Normalize column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip().lower() for col in df.columns.values]
    else:
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    df["source"] = label
    return df


def save_data_to_csv():
    fixtures_url = (
        "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    )
    shooting_url = "https://fbref.com/en/comps/9/shooting/Premier-League-Stats"

    matches = fetch_table(fixtures_url, "sched_2024-2025_9_1", "matches")
    shooting_for = fetch_table(
        shooting_url, "stats_squads_shooting_for", "shooting_for"
    )
    shooting_against = fetch_table(
        shooting_url, "stats_squads_shooting_against", "shooting_against"
    )

    if matches is not None:
        matches.to_csv("premier_league_matches.csv", index=False)
        print("Saved matches data to 'premier_league_matches.csv'.")

    if shooting_for is not None and shooting_against is not None:
        shooting_data = pd.concat([shooting_for, shooting_against], ignore_index=True)
        shooting_data.to_csv("premier_league_shooting_stats.csv", index=False)
        print("Saved shooting data to 'premier_league_shooting_stats.csv'.")
