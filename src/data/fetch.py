import os
import time
import re
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException
import pandas as pd
from bs4 import BeautifulSoup
from config.leagues import LEAGUES

# Single global driver instance reused per league
_driver = None


def get_driver():
    global _driver
    if _driver is None:
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
        _driver = webdriver.Chrome(service=service, options=options)
        _driver.set_page_load_timeout(120)
    return _driver


def close_driver():
    global _driver
    if _driver:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None


class SeleniumResponse:
    def __init__(self, text):
        self.text = text


def get_current_season(base_url):
    """
    Henter gjeldende sesong ved å lese <div id="meta"><h1>2024-2025 Premier League Stats</h1></div>.
    Returnerer sesong som '2024-2025', eller None hvis ikke funnet.
    """
    resp = fetch_data(base_url)
    if not resp:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    h1 = soup.select_one("div#meta h1")
    if not h1:
        return None

    text = h1.get_text(strip=True)
    m = re.match(r"^(\d{4}-\d{4})", text)
    return m.group(1) if m else None


def get_prev_season(season):
    """
    Gitt "2023-2024", returnerer "2022-2023".
    """
    start, end = season.split("-")
    return f"{int(start)-1}-{int(end)-1}"


def fetch_data(url):
    """
    Fetch page HTML using a shared Selenium WebDriver. Retries on failure with a delay.
    """
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(2)
        html = driver.page_source
        return SeleniumResponse(html)
    except WebDriverException as e:
        print(f"WebDriver error: {e}. Retrying in 60 seconds...")
        close_driver()
        time.sleep(60)
        return fetch_data(url)
    except Exception as e:
        # Fanger opp f.eks. HTTPConnectionPool–feil mot ChromeDriver og andre uventede feil
        print(f"Fetch data error: {e}. Retrying in 60 seconds...")
        close_driver()
        time.sleep(60)
        return fetch_data(url)


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
    Hent alle kamper (inkl. kommende) fra div_matchlogs_for, og shooting-stats.
    """
    data = fetch_data(team_url)
    if not data:
        return None
    soup = BeautifulSoup(data.text, "lxml")

    # 1) Metadata
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

    df = df_meta.merge(df_for_shoot, on="date", how="left").merge(
        df_against_shoot, on="date", how="left"
    )
    return df


def fetch_league_data(league_name, cfg):
    comp_id = cfg["comp_id"]
    slug = cfg["slug"]
    base_url = f"https://fbref.com/en/comps/{comp_id}/{slug}"
    current = get_current_season(base_url)
    prev = get_prev_season(current) if current else None

    print(f"\n--- Henter data for liga: {league_name} ---")
    all_data = []
    standings_urls = [
        (season, f"https://fbref.com/en/comps/{comp_id}/{season}/{season}-{slug}")
        for season in filter(None, (prev, current))
    ]

    for season, standings_url in standings_urls:
        print(f"\n--- Henter {league_name}, sesong {season} ---")
        team_urls = fetch_team_urls(standings_url)
        for team_url in team_urls:
            attempts = 0
            df_team = None
            while attempts < 3 and df_team is None:
                try:
                    df_team = fetch_team_data(team_url, season=season)
                except Exception as e:
                    attempts += 1
                    print(f"Error fetching {team_url} (attempt {attempts}): {e}")
                    close_driver()
                    time.sleep(5 * attempts)
                if df_team is None and attempts < 3:
                    time.sleep(5)
            if df_team is None:
                print(f"Skipping {team_url} after 3 failed attempts")
            else:
                all_data.append(df_team)

    close_driver()

    if not all_data:
        print(f"Ingen kamper samlet for liga {league_name}, hopper over oppdatering.")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    # Velg relevante kolonner
    want = [
        "date",
        "time",
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

    raw_file = os.path.join(
        "data",
        "raw",
        f"{league_name.lower().replace(' ', '_')}_matches_full.csv",
    )
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)

    if prev:
        seasons_fetched = df_final["season"].unique().tolist()
        if prev not in seasons_fetched and os.path.exists(raw_file):
            print(f"Sesong {prev} feilet, legger til gammel data fra {raw_file}")
            old = pd.read_csv(raw_file)
            prev_old = old[old["season"] == prev]
            if not prev_old.empty:
                df_final = pd.concat([df_final, prev_old], ignore_index=True)

    df_final.to_csv(raw_file, index=False)
    print(f"Lagret {raw_file}")


def main():
    for league_name, cfg in LEAGUES.items():
        fetch_league_data(league_name, cfg)


if __name__ == "__main__":
    main()
