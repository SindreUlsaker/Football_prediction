import os
import time
import re
import random
from io import StringIO
from selenium import webdriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
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
    """
    Returnerer en Chrome-driver. Prøver først undetected-chromedriver (UC) med stealth.
    Hvis UC ikke er installert/feiler, faller vi tilbake til standard Selenium Chrome.
    """
    global _driver
    if _driver is not None:
        return _driver

    # Prøv å importere UC inni funksjonen (ikke på toppnivå)
    uc = None
    try:
        import undetected_chromedriver as _uc

        uc = _uc
    except Exception as e:
        print(
            f"[fetch] undetected-chromedriver ikke tilgjengelig ({e}). Faller tilbake til standard Selenium.",
            flush=True,
        )

    if uc:
        # --- UC driver med stealth ---
        ua = os.getenv(
            "FBREF_UA",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )
        lang = os.getenv("FBREF_LANG", "en-US,en;q=0.9")
        proxy = os.getenv("FBREF_PROXY")  # f.eks. http://user:pass@host:port

        options = uc.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--blink-settings=imagesEnabled=true")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--lang=" + lang)
        if proxy:
            options.add_argument(f"--proxy-server={proxy}")

        _driver = uc.Chrome(options=options)
        _driver.set_page_load_timeout(60)
        _driver.set_script_timeout(30)

        # Stealth: skjul webdriver + sett UA via CDP (best effort)
        try:
            _driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                        Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});
                        Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
                    """
                },
            )
            _driver.execute_cdp_cmd(
                "Network.setUserAgentOverride",
                {"userAgent": ua, "acceptLanguage": lang, "platform": "Windows"},
            )
        except Exception:
            pass

        # Warm-up: seed cookies
        try:
            _driver.get("https://fbref.com/")
            time.sleep(1.2 + random.random())
        except Exception:
            pass

    else:
        # --- Standard Selenium Chrome ---
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--blink-settings=imagesEnabled=true")
        options.add_experimental_option(
            "excludeSwitches", ["enable-automation", "enable-logging"]
        )
        options.add_experimental_option("useAutomationExtension", False)
        options.page_load_strategy = "eager"

        _driver = webdriver.Chrome(options=options)
        _driver.set_page_load_timeout(60)
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


def compute_local_current_season(tz: str = "Europe/Oslo") -> str:
    """
    Lokal fallback for sesongstreng når scraping feiler:
      - Aug (8)–Des (12): YYYY-(YYYY+1)
      - Jan (1)–Jul (7): (YYYY-1)-YYYY
    """
    now = pd.Timestamp.now(tz=tz)
    y = now.year
    if 8 <= now.month <= 12:
        return f"{y}-{y+1}"
    else:
        return f"{y-1}-{y}"


def get_current_season(base_url: str) -> str:
    """
    Hent gjeldende sesong fra <div id="meta"><h1>…</h1></div>.
    Fallback til lokal beregning (compute_local_current_season) hvis parsing feiler
    eller h1 mangler.
    """
    resp = fetch_data(base_url)
    if resp:
        soup = BeautifulSoup(resp.text, "lxml")
        h1 = soup.select_one("div#meta h1")
        if h1:
            txt = h1.get_text(strip=True)
            m = re.match(r"^(\d{4}-\d{4})", txt)
            if m:
                return m.group(1)

    # Fallback når scraping ikke gir treff:
    return compute_local_current_season()


def get_prev_season(season):
    """
    Gitt "2023-2024", returnerer "2022-2023".
    """
    start, end = season.split("-")
    return f"{int(start)-1}-{int(end)-1}"


def fetch_data(url):
    """
    Hent HTML med retries/backoff. Detekter Cloudflare/interstitial både via
    kjente markører og mistenkelig lav HTML-lengde, og restart driver når det skjer.
    Støtter proxy/UA/stealth via get_driver().
    """
    global _last_challenge_ts

    # Cooldown hvis vi nettopp traff challenge
    since = time.time() - _last_challenge_ts
    if since < 45:
        wait = int(45 - since) + 1
        print(f"[fetch] Cooldown {wait}s pga. nylig challenge", flush=True)
        time.sleep(wait)

    max_attempts = 8  # litt høyere når vi bruker UC
    attempt = 0
    start = time.time()

    CHALLENGE_MARKERS = [
        "Attention Required",
        "cf-browser-verification",
        "cf-challenge",
        "Why do I have to complete a CAPTCHA",
        "Just a moment",
        "Checking your browser",
        "/cdn-cgi/challenge-platform/",
    ]
    # Stats-/lag-sider hos fbref er normalt store. Sett konservativ terskel.
    MIN_OK_HTML_LEN = int(os.getenv("FBREF_MIN_HTML", "80000"))

    def looks_like_challenge(html: str) -> bool:
        if not html:
            return True
        lower = html.lower()
        if any(m.lower() in lower for m in CHALLENGE_MARKERS):
            return True
        if len(html) < MIN_OK_HTML_LEN:
            return True
        return False

    while attempt < max_attempts:
        attempt += 1
        try:
            driver = get_driver()
            print(f"[fetch] GET {url} (attempt {attempt})", flush=True)
            driver.get(url)
            time.sleep(1.0 + random.random())
            html = driver.page_source
            print(f"[fetch] Hentet HTML-lengde: {len(html)}", flush=True)

            if looks_like_challenge(html):
                _last_challenge_ts = time.time()
                print(
                    "[fetch] Mistenkt challenge/interstitial. Restart driver + backoff...",
                    flush=True,
                )
                close_driver()
                # Økende backoff med litt jitter
                sleep_s = 10 + 6 * attempt + random.uniform(0, 3)
                time.sleep(sleep_s)
                continue

            return SeleniumResponse(html)

        except TimeoutException as e:
            print(f"[fetch] Timeout on {url}: {e}. Restart + backoff...", flush=True)
            close_driver()
            time.sleep(5 * attempt)
            continue
        except WebDriverException as e:
            print(f"[fetch] WebDriver error on {url}: {e}. Restart...", flush=True)
            close_driver()
            time.sleep(5 * attempt)
            continue
        except Exception as e:
            print(f"[fetch] Unexpected error on {url}: {e}. Restart...", flush=True)
            close_driver()
            time.sleep(5 * attempt)
            continue
        finally:
            if time.time() - start > 300:  # maks ~5 min per URL
                print(
                    f"[fetch] Giving up on {url} etter ~5 minutter total.", flush=True
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
    df_for_safe = pd.DataFrame({"date": df_for["date"]})
    for col in shoot_cols:
        if col in df_for.columns:
            df_for_safe[f"{col}_for"] = df_for[col]
        else:
            df_for_safe[f"{col}_for"] = ""

    # Lag safe DataFrame for "against"-statistikk
    df_against_safe = pd.DataFrame({"date": df_against["date"]})
    for col in shoot_cols:
        if col in df_against.columns:
            df_against_safe[f"{col}_against"] = df_against[col]
        else:
            df_against_safe[f"{col}_against"] = ""

    df = df_meta.merge(df_for_safe, on="date", how="left").merge(
        df_against_safe, on="date", how="left"
    )
    return df


# --- NY: Hent promoted-lag og Pts/MP fra nivå 2-tabellen for en gitt (prev) sesong ---
def fetch_second_division_promoted(
    league_cfg: dict, season: str
) -> dict[tuple[str, str], float]:
    """
    Returnerer {(season, team_mapped): pts_per_match} for lag merket 'Promoted' i nivå 2-tabellen.
    - league_cfg må inneholde 'comp_id_second_division' og 'slug_second_division' og ev. 'team_name_map'.
    - season er forrige sesong-strengen, f.eks. '2023-2024'.
    """
    comp2 = league_cfg.get("comp_id_second_division")
    slug2 = league_cfg.get("slug_second_division")
    if not comp2 or not slug2:
        return {}

    url = f"https://fbref.com/en/comps/{comp2}/{season}/{season}-{slug2}"
    resp = fetch_data(url)
    if not resp:
        return {}

    soup = BeautifulSoup(resp.text, "lxml")

    # Finn tabell med id som inneholder "results"
    target_table = None
    for tbl in soup.find_all("table"):
        tbl_id = tbl.get("id", "")
        if "results" in tbl_id.lower():
            target_table = tbl
            break
    if target_table is None:
        return {}

    df = pd.read_html(StringIO(str(target_table)))[0]

    # Normaliser kolonnenavn
    df.columns = [str(c).strip() for c in df.columns]
    cols = df.columns

    # Finn promoted-rader
    notes_col = None
    for c in cols:
        if str(c).lower().startswith("notes"):
            notes_col = c
            break
    if notes_col is None:
        return {}

    promoted = df[
        df[notes_col].astype(str).str.contains("Promoted", case=False, na=False)
    ].copy()
    if promoted.empty:
        return {}

    # Finn lagnavn- og pts/mp-kolonner (fall back til Pts og MP)
    squad_col = None
    for c in cols:
        if str(c).lower() in ("squad", "team"):
            squad_col = c
            break
    if squad_col is None:
        return {}

    if "Pts/MP" in cols:
        promoted["pts_mp"] = pd.to_numeric(promoted["Pts/MP"], errors="coerce")
    else:
        try:
            promoted["pts_mp"] = pd.to_numeric(
                promoted["Pts"], errors="coerce"
            ) / pd.to_numeric(promoted["MP"], errors="coerce")
        except Exception:
            return {}

    team_map = league_cfg.get("team_name_map") or {}
    out = {}
    for _, r in promoted.iterrows():
        raw_name = str(r[squad_col]).strip()
        mapped = team_map.get(raw_name, raw_name)
        val = float(r["pts_mp"]) if pd.notna(r["pts_mp"]) else None
        if val is not None and val > 0:
            out[(season, mapped)] = val
    return out


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
