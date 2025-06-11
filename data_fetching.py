import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from io import StringIO

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://google.com",
    }
)

def fetch_data(url):
    try:
        response = session.get(url)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            print("Too many requests. Waiting before retrying...")
            time.sleep(60)
            return fetch_data(url)
        else:
            print(f"HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def fetch_table(url, table_id, label):
    data = fetch_data(url)
    if not data:
        return None

    soup = BeautifulSoup(data.text, 'lxml')
    table = soup.find('table', {'class': 'stats_table', 'id': table_id})
    if not table:
        print(f"Could not find the table with id '{table_id}'.")
        return None

    df = pd.read_html(StringIO(str(table)))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip().lower() for col in df.columns.values]
    else:
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    df['source'] = label
    return df

def save_data_to_csv():
    fixtures_url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    shooting_url = "https://fbref.com/en/comps/9/shooting/Premier-League-Stats"

    matches = fetch_table(fixtures_url, "sched_2024-2025_9_1", "matches")
    shooting_for = fetch_table(shooting_url, "stats_squads_shooting_for", "shooting_for")
    shooting_against = fetch_table(shooting_url, "stats_squads_shooting_against", "shooting_against")

    if matches is not None:
        matches.to_csv("premier_league_matches.csv", index=False)
        print("Saved matches data to 'premier_league_matches.csv'.")

    if shooting_for is not None and shooting_against is not None:
        shooting_data = pd.concat([shooting_for, shooting_against], ignore_index=True)
        shooting_data.to_csv("premier_league_shooting_stats.csv", index=False)
        print("Saved shooting data to 'premier_league_shooting_stats.csv'.")
        
    
