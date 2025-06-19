import os
import time
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException
import pandas as pd
from bs4 import BeautifulSoup
from config.leagues import LEAGUES


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


def fetch_team_urls(standings_url):
    data = fetch_data(standings_url)
    if not data:
        return []
    soup = BeautifulSoup(data.text, "lxml")
    table = soup.select_one("table.stats_table")
    if not table:
        return []
    return [
        f"https://fbref.com{a['href']}"
        for a in table.find_all("a", href=True)
        if "/squads/" in a["href"]
    ]


def fetch_team_data(team_url, season="2024-2025"):  # fmt: skip
    """
    Hent alle kamper (inkl. kommende) fra div_matchlogs_for på lagets Stats-side,
    og legg til shooting-stats for spilte kamper.
    """
    data = fetch_data(team_url)
    if not data:
        return None
    soup = BeautifulSoup(data.text, "lxml")

    # 1) Metadata: date, time, comp, opponent, venue, round
    div_meta = soup.find("div", id="div_matchlogs_for")
    if not div_meta:
        print(f"Fant ingen matchlogs_for for {team_url}")
        return None
    tbl_meta = div_meta.find("table")
    df_meta = pd.read_html(StringIO(str(tbl_meta)))[0]
    if isinstance(df_meta.columns, pd.MultiIndex):
        df_meta.columns = [
            "_".join(col).strip().lower() for col in df_meta.columns.values
        ]
    else:
        df_meta.columns = [c.lower().replace(" ", "_") for c in df_meta.columns]
    meta_cols = ["date", "time", "comp", "opponent", "venue", "round"]
    existing_meta = [c for c in meta_cols if c in df_meta.columns]
    df_meta = df_meta[existing_meta].copy()
    df_meta["season"] = season
    team_slug = team_url.rstrip("/").split("/")[-1]
    team = team_slug.replace("-Stats", "").replace("-", " ")
    df_meta["team"] = team.strip()

    # 2) Shooting stats
    shooting_a = soup.find("a", string="Shooting")
    if not shooting_a:
        return df_meta
    shooting_url = f"https://fbref.com{shooting_a['href']}"
    shoot_page = fetch_data(shooting_url)
    if not shoot_page:
        return df_meta
    soup2 = BeautifulSoup(shoot_page.text, "lxml")
    div_for = soup2.find("div", id="div_matchlogs_for")
    div_against = soup2.find("div", id="div_matchlogs_against")
    if not div_for or not div_against:
        return df_meta
    tbl_for = div_for.find("table")
    tbl_against = div_against.find("table")
    df_for = pd.read_html(StringIO(str(tbl_for)))[0]
    df_against = pd.read_html(StringIO(str(tbl_against)))[0]
    for df in (df_for, df_against):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    shoot_cols = ["xg", "gf", "ga", "sh", "sot", "dist", "fk", "pk"]
    df_for_shoot = df_for[["date"] + shoot_cols].rename(
        columns={c: f"{c}_for" for c in shoot_cols}
    )
    df_against_shoot = df_against[["date"] + shoot_cols].rename(
        columns={c: f"{c}_against" for c in shoot_cols}
    )

    # Merge meta with shooting stats
    df = df_meta.merge(df_for_shoot, on="date", how="left").merge(
        df_against_shoot, on="date", how="left"
    )
    return df


def fetch_league_data(league_name, standings_url, season="2024-2025"):  # fmt: skip
    """
    Hent og lagre data for en enkelt liga.
    """
    print(f"\n--- Henter data for liga: {league_name} ---")
    team_urls = fetch_team_urls(standings_url)
    if not team_urls:
        print(f"Ingen lag funnet for liga {league_name}.")
        return

    all_data = []
    for url in team_urls:
        print(f"Henter data for {url}")
        df_team = fetch_team_data(url, season)
        time.sleep(1)
        if df_team is not None:
            all_data.append(df_team)

    if not all_data:
        print(f"Ingen kamper samlet for liga {league_name}.")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    # Velg relevante kolonner
    want = [
        "date",
        "comp",
        "season",
        "team",
        "opponent",
        "venue",
        "round",
        "xg_for",
        "gf_for",
        "ga_for",
        "sh_for",
        "sot_for",
        "dist_for",
        "fk_for",
        "pk_for",
        "xg_against",
        "gf_against",
        "ga_against",
        "sh_against",
        "sot_against",
        "dist_against",
        "fk_against",
        "pk_against",
    ]
    existing = [c for c in want if c in df_all.columns]
    df_final = df_all[existing]

    # Lagre per liga i egen CSV
    filename = os.path.join(
        "data",
        "raw",
        league_name.lower().replace(" ", "_") + "_matches_full.csv",
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_final.to_csv(filename, index=False)
    print(f"Lagret {filename}")


def main():
    # Iterer gjennom alle ligaer definert i config/leagues.py
    for league_name, cfg in LEAGUES.items():
        fetch_league_data(
            league_name,
            cfg["standings_url"],
            season=cfg.get("season", "2024-2025"),
        )


if __name__ == "__main__":
    main()
