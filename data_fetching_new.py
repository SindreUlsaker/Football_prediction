import time
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

from fetch_data_selenium import fetch_data  # Selenium-henting


def fetch_team_urls(standings_url):
    """Hent alle lag‐URLer fra Premier League-standings."""
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
    # 1) Åpne lagets Stats-side
    data = fetch_data(team_url)
    if not data:
        return None
    soup = BeautifulSoup(data.text, "lxml")

    # 2) Hent scores & fixtures fra <div id="div_matchlogs_for">
    div_meta = soup.find("div", id="div_matchlogs_for")
    if not div_meta:
        print(f"Fant ingen matchlogs_for for {team_url}")
        return None
    tbl_meta = div_meta.find("table")
    df_meta = pd.read_html(StringIO(str(tbl_meta)))[0]
    # Rydd kolonnenavn
    if isinstance(df_meta.columns, pd.MultiIndex):
        df_meta.columns = [
            "_".join(col).strip().lower() for col in df_meta.columns.values
        ]
    else:
        df_meta.columns = [c.lower().replace(" ", "_") for c in df_meta.columns]
    # Velg metadata-kolonner for alle kamper
    meta_cols = ["date", "comp", "opponent", "venue", "round"]
    existing_meta = [c for c in meta_cols if c in df_meta.columns]
    df_meta = df_meta[existing_meta].copy()
    df_meta["season"] = season
    team = team_url.rstrip("/").split("/")[-1].replace("-Stats", " ").replace("-", " ")
    df_meta["team"] = team

    # 3) Hent shooting-stats fra shooting-siden
    shooting_a = soup.find("a", string="Shooting")
    if not shooting_a:
        # ingen shooting-tabell, return kun meta
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

    # 4) Rydd kolonner for shooting-dfs
    for df in (df_for, df_against):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # 5) Velg kun shooting-stats
    shoot_cols = ["xg", "gf", "ga", "sh", "sot", "dist", "fk", "pk"]
    # prefix kun shoot-stats
    df_for_shoot = df_for[["date"] + shoot_cols].rename(
        columns={c: f"{c}_for" for c in shoot_cols}
    )
    df_against_shoot = df_against[["date"] + shoot_cols].rename(
        columns={c: f"{c}_against" for c in shoot_cols}
    )

    # 6) Merge metadata med shoot-stats med left join
    df = df_meta.merge(df_for_shoot, on="date", how="left").merge(
        df_against_shoot, on="date", how="left"
    )
    return df


def main():
    standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    team_urls = fetch_team_urls(standings_url)
    if not team_urls:
        print("Ingen lag funnet.")
        return

    all_data = []
    for url in team_urls:
        print(f"Henter data for {url}")
        df_team = fetch_team_data(url)
        time.sleep(1)
        if df_team is not None:
            all_data.append(df_team)

    if not all_data:
        print("Ingen kamper samlet.")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    # velg ønskede kolonner
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

    df_final.to_csv("premier_league_matches_full.csv", index=False)
    print("Lagret premier_league_matches_full.csv")


if __name__ == "__main__":
    main()
