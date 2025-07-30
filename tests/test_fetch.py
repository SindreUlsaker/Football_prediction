import pytest
import pandas as pd
from io import StringIO

from src.data import fetch
from src.data.fetch import (
    get_prev_season,
    get_current_season,
    fetch_team_urls,
    fetch_team_data,
    SeleniumResponse,
)

# --- get_prev_season ---


def test_get_prev_season_valid():
    assert get_prev_season("2023-2024") == "2022-2023"


def test_get_prev_season_invalid_format():
    with pytest.raises(ValueError):
        get_prev_season("20232024")  # mangler '-'


# --- get_current_season ---


def test_get_current_season_success(monkeypatch):
    html = '<div id="meta"><h1>2025-2026 Premier League Stats</h1></div>'
    monkeypatch.setattr(fetch, "fetch_data", lambda url: SeleniumResponse(html))
    assert get_current_season("any_url") == "2025-2026"


def test_get_current_season_no_response(monkeypatch):
    monkeypatch.setattr(fetch, "fetch_data", lambda url: None)
    assert get_current_season("any_url") is None


def test_get_current_season_no_h1(monkeypatch):
    monkeypatch.setattr(
        fetch, "fetch_data", lambda url: SeleniumResponse("<div></div>")
    )
    assert get_current_season("any_url") is None


# --- fetch_team_urls ---


def test_fetch_team_urls_success(monkeypatch):
    html = """
    <table class="stats_table">
      <tr><td><a href="/en/squads/team1-Stats">Team1</a></td></tr>
      <tr><td><a href="/en/squads/team2-Stats">Team2</a></td></tr>
    </table>
    """
    monkeypatch.setattr(fetch, "fetch_data", lambda url: SeleniumResponse(html))
    urls = fetch_team_urls("dummy")
    assert urls == [
        "https://fbref.com/en/squads/team1-Stats",
        "https://fbref.com/en/squads/team2-Stats",
    ]


def test_fetch_team_urls_no_data(monkeypatch):
    monkeypatch.setattr(fetch, "fetch_data", lambda url: None)
    assert fetch_team_urls("dummy") == []


def test_fetch_team_urls_no_table(monkeypatch):
    monkeypatch.setattr(
        fetch, "fetch_data", lambda url: SeleniumResponse("<html></html>")
    )
    assert fetch_team_urls("dummy") == []


# --- fetch_team_data ---


def test_fetch_team_data_meta_only(monkeypatch):
    meta_html = """
    <div id="div_matchlogs_for">
      <table>
        <tr><th>Date</th><th>Time</th><th>Comp</th><th>Opponent</th><th>Venue</th><th>Round</th></tr>
        <tr><td>2025-07-29</td><td>20:00</td><td>PL</td><td>Team B</td><td>Home</td><td>Matchweek 1</td></tr>
      </table>
    </div>
    """
    monkeypatch.setattr(fetch, "fetch_data", lambda url: SeleniumResponse(meta_html))
    df = fetch_team_data("https://fbref.com/team-Stats", season="2025-2026")
    assert df.shape == (1, len(df.columns))
    assert df["comp"].iloc[0] == "PL"
    assert df["team"].iloc[0] == "team"


def test_fetch_team_data_with_shooting(monkeypatch):
    meta_html = """
    <html><body>
      <div id="div_matchlogs_for">
        <table>
          <tr><th>Date</th><th>Time</th><th>Comp</th><th>Opponent</th><th>Venue</th><th>Round</th></tr>
          <tr><td>2025-07-30</td><td>18:00</td><td>PL</td><td>Team C</td><td>Away</td><td>Matchweek 2</td></tr>
        </table>
      </div>
      <a href="/shooting-url">Shooting</a>
    </body></html>
    """
    shoot_html = """
    <html><body>
      <div id="div_matchlogs_for">
        <table>
          <tr>
            <th>Date</th><th>xG</th><th>GF</th><th>GA</th><th>Sh</th><th>Sot</th><th>Dist</th><th>FK</th><th>PK</th>
          </tr>
          <tr>
            <td>2025-07-30</td><td>1.1</td><td>2</td><td>0</td><td>5</td><td>3</td><td>30</td><td>1</td><td>0</td>
          </tr>
        </table>
      </div>
      <div id="div_matchlogs_against">
        <table>
          <tr>
            <th>Date</th><th>xG</th><th>GF</th><th>GA</th><th>Sh</th><th>Sot</th><th>Dist</th><th>FK</th><th>PK</th>
          </tr>
          <tr>
            <td>2025-07-30</td><td>0.9</td><td>1</td><td>3</td><td>4</td><td>2</td><td>25</td><td>0</td><td>1</td>
          </tr>
        </table>
      </div>
    </body></html>
    """

    def fake_fetch(url, *args, **kwargs):
        if "shooting-url" in url:
            return SeleniumResponse(shoot_html)
        return SeleniumResponse(meta_html)

    monkeypatch.setattr(fetch, "fetch_data", fake_fetch)
    df = fetch_team_data("https://fbref.com/team-Stats", season="2025-2026")

    # Forventede shooting-kolonner
    for col in [
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
    ]:
        assert col in df.columns

    assert df["xg_for"].iloc[0] == 1.1
    assert df["pk_against"].iloc[0] == 1
