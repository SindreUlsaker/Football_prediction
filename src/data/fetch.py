import os
import time
import re
import random
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import TimeoutException
import pandas as pd
from bs4 import BeautifulSoup
from config.leagues import LEAGUES
from config.settings import DATA_PATH
import argparse
import itertools

# Single global driver instance reused per league
_driver = None
_last_challenge_ts = 0

def get_driver():
    global _driver
    if _driver is None:
        options = Options()
        # Headless-støtte i moderne Chrome
        options.add_argument("--headless=new")
        # Kritisk i GitHub Actions (container)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        # Stabilitet/perf i CI
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument(
            "--disable-features=Translate,BackForwardCache,AutomationControlled"
        )
        options.add_argument("--blink-settings=imagesEnabled=true")
        # Fjern gammel, mistenkelig UA (Chrome 91). La Chrome bruke sin egen UA.
        # Hvis du MÅ sette UA: sett en moderne (>= 120) – men start uten.

        # Mindre "bot-støy":
        options.add_experimental_option(
            "excludeSwitches", ["enable-automation", "enable-logging"]
        )
        options.add_experimental_option("useAutomationExtension", False)

        # Lastestrategi: ikke vent på alt (reduserer heng i CI)
        options.page_load_strategy = "eager"

        # Bruk Selenium Manager (matcher chromedriver automatisk)
        _driver = webdriver.Chrome(options=options)

        # Stramme timeouts
        _driver.set_page_load_timeout(60)  # var 120
        _driver.set_script_timeout(30)

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
    Fetch page HTML with Selenium. Oppdager Cloudflare 'challenge' og tar en
    global cooldown slik at resten av lagene ikke feiler i serie.
    """
    global _last_challenge_ts

    # Hvis vi nylig traff en challenge, vent litt før vi prøver neste side
    since = time.time() - _last_challenge_ts
    if since < 45:  # 45s "cooldown"-vindu
        wait = int(45 - since) + 1
        print(f"[fetch] Cooldown {wait}s pga. nylig challenge", flush=True)
        time.sleep(wait)

    max_attempts = 4
    attempt = 0
    start = time.time()

    while attempt < max_attempts:
        attempt += 1
        try:
            driver = get_driver()
            print(f"[fetch] GET {url} (attempt {attempt})", flush=True)
            driver.get(url)
            time.sleep(1.0 + random.random())
            html = driver.page_source

            # Cloudflare / challenge heuristikk
            if (
                "Attention Required" in html
                or "cf-browser-verification" in html
                or "cf-challenge" in html
                or "Why do I have to complete a CAPTCHA" in html
            ):
                _last_challenge_ts = time.time()
                print(
                    "[fetch] Cloudflare challenge oppdaget. Restart driver + backoff...",
                    flush=True,
                )
                close_driver()
                time.sleep(20 * attempt)  # økende backoff
                continue

            return SeleniumResponse(html)

        except TimeoutException as e:
            print(
                f"[fetch] Timeout on {url}: {e}. Restarting driver og backoff...",
                flush=True,
            )
            close_driver()
            time.sleep(5 * attempt)
            continue
        except WebDriverException as e:
            print(
                f"[fetch] WebDriver error on {url}: {e}. Restarting driver...",
                flush=True,
            )
            close_driver()
            time.sleep(5 * attempt)
            continue
        except Exception as e:
            print(
                f"[fetch] Unexpected error on {url}: {e}. Restarting driver...",
                flush=True,
            )
            close_driver()
            time.sleep(5 * attempt)
            continue
        finally:
            # Ikke la én URL spise hele jobben
            if time.time() - start > 180:
                print(
                    f"[fetch] Giving up on {url} etter ~3 minutter total.", flush=True
                )
                break

    return None


def fetch_team_urls(standings_url):
    data = fetch_data(standings_url)
    if not data:
        return []
    soup = BeautifulSoup(data.text, "lxml")
    table = soup.select_one("table.stats_table")
    if not table:
        return []
    urls = []
    for a in table.find_all("a", href=True):
        href = a["href"]
        # Kun rene lag-Stats-sider, unngå matchlogs etc.
        if "/squads/" in href and "matchlogs" not in href and href.endswith("-Stats"):
            urls.append(f"https://fbref.com{href}")
    print(f"[fetch] Fant {len(urls)} lag i {standings_url}", flush=True)
    return urls


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

    shoot_cols_for = ["xg_for","gf_for","ga_for","sh_for","sot_for","dist_for","fk_for","pk_for"]
    shoot_cols_against = ["xg_against","gf_against","ga_against","sh_against","sot_against","dist_against","fk_against","pk_against"]

    shooting_a = soup.find("a", string="Shooting")
    if not shooting_a:
        for col in itertools.chain(shoot_cols_for, shoot_cols_against):
            df_meta[col] = ""
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


def fetch_league_data(league_name, cfg, seasons_to_fetch=None):
    comp_id = cfg["comp_id"]
    slug = cfg["slug"]
    base_url = f"https://fbref.com/en/comps/{comp_id}/{slug}"

    # Hvis ingen sesonger er spesifisert: hent prev + current automatisk
    if seasons_to_fetch is None:
        current = get_current_season(base_url)
        prev = get_prev_season(current) if current else None
        seasons_to_fetch = list(filter(None, [prev, current]))

    print(f"\n--- Henter data for liga: {league_name} ---")
    all_data = []
    standings_urls = [
        (season, f"https://fbref.com/en/comps/{comp_id}/{season}/{season}-{slug}")
        for season in seasons_to_fetch
    ]

    for season, standings_url in standings_urls:
        print(f"\n--- Henter {league_name}, sesong {season} ---")
        team_urls = fetch_team_urls(standings_url)

        fail_streak = 0  # teller hvor mange lag på rad som mangler data

        for team_url in team_urls:
            attempts = 0
            df_team = None

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
                print(f"[fetch] Skipping {team_url} after 3 failed attempts")
                fail_streak += 1
            else:
                all_data.append(df_team)
                fail_streak = 0  # nullstill streak ved suksess

            # --- anti-blokk logikk ---
            time.sleep(random.uniform(3, 5))  # liten pause etter hvert lag
            if fail_streak >= 3:
                print(
                    "[fetch] 3 lag på rad uten matchlogs_for – mistenker blokkering, tar 2 minutters pause og restart av driver..."
                )
                close_driver()
                time.sleep(120)
                fail_streak = 0

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

    league_key = league_name.lower().replace(" ", "_")

    raw_dir = os.path.join(DATA_PATH, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    raw_file = os.path.join(raw_dir, f"{league_key}_{season}_matches.csv")

    df_final.to_csv(raw_file, index=False)
    print(f"[INFO] Skrev {len(df_final)} rader til {os.path.abspath(raw_file)}")


def main(seasons=None):
    parser = argparse.ArgumentParser(
        description="Fetch match data for configured leagues"
    )
    parser.add_argument(
        "-s",
        "--seasons",
        nargs="+",
        help="List of seasons to fetch, f.eks. 2024-2025 eller 2023-2024 2024-2025",
    )
    args = parser.parse_args()

    # args.seasons har høyeste prioritet, ellers bruk seasons-argumentet
    seasons_to_fetch = args.seasons or seasons

    for league_name, cfg in LEAGUES.items():
        fetch_league_data(league_name, cfg, seasons_to_fetch)


if __name__ == "__main__":
    main()
