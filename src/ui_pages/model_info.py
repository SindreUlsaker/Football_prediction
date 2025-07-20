import streamlit as st
import pandas as pd
from config.leagues import LEAGUES
from config.settings import DATA_PATH
from src.models.predict import load_models_for_league


def show_feature_weights(model, feature_names):
    """Viser kun relativ betydning av hver generisk feature i prosent."""

    # Brukervennlige navn for de generiske feature-keys
    name_map = {
        "xg_roll5": "xG siste 5 kamper (lag)",
        "xg_roll10": "xG siste 10 kamper (lag)",
        "gf_roll5": "Mål scoret siste 5 kamper (lag)",
        "gf_roll10": "Mål scoret siste 10 kamper (lag)",
        "avg_goals_for": "Gj.snittsmål scoret (lag)",
        "xg_conceded_away_roll5": "xG motstander siste 5 kamper",
        "xg_conceded_away_roll10": "xG motstander siste 10 kamper",
        "ga_away_roll5": "Inslupne mål motstander siste 5 kamper",
        "ga_away_roll10": "Inslupne mål motstander siste 10 kamper",
        "avg_goals_against_away": "Gj.snitts mål sluppet inn av motstander",
        "is_home": "Spiller på hjemmebane?",
    }

    # Hvilke generiske features vi ønsker å vise
    team_features = [
        "xg_roll5",
        "xg_roll10",
        "gf_roll5",
        "gf_roll10",
        "avg_goals_for",
        "is_home",
    ]
    opponent_features = [
        "xg_conceded_away_roll5",
        "xg_conceded_away_roll10",
        "ga_away_roll5",
        "ga_away_roll10",
        "avg_goals_against_away",
    ]

    # 1) Hent alle koeffisienter fra modellen
    coefs = model.coef_.flatten()

    # 2) Sørg for at 'is_home' er med i feature_names
    if "is_home" not in feature_names:
        feature_names = feature_names + ["is_home"]

    # 3) Ta akkurat like mange coefs som det er generiske features
    generic_coefs = coefs[: len(feature_names)]

    # 4) Bygg DataFrame kun over de generiske features
    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Weight": generic_coefs,
        }
    )
    df["AbsWeight"] = df["Weight"].abs()
    df["Navn"] = df["Feature"].map(name_map).fillna(df["Feature"])
    total = df["AbsWeight"].sum()
    df["Betydning (%)"] = (df["AbsWeight"] / total * 100).round(1)

    # 5) Vis resultatene i to tabeller
    st.subheader("📊 Relativ feature‐betydning")

    st.markdown("**Lagets egne features**")
    df_team = df[df["Feature"].isin(team_features)][
        ["Navn", "Betydning (%)"]
    ].reset_index(drop=True)
    st.dataframe(df_team, hide_index=True, use_container_width=True)

    st.markdown("**Motstander‐features**")
    df_opp = df[df["Feature"].isin(opponent_features)][
        ["Navn", "Betydning (%)"]
    ].reset_index(drop=True)
    st.dataframe(df_opp, hide_index=True, use_container_width=True)


def show_model_info_page():
    """
    Side for å forklare hvordan modellen fungerer, vise disclaimer,
    og presentere relative feature-vekter per liga.
    """
    st.header("Om modellen 📊")
    st.markdown(
        """
        Modellen bruker statistikk fra tidligere kamper til å estimere hvor mange mål hvert lag kommer til å score. 
        Til dette benyttes blant annet xG, scorede mål og innslupne mål, både for laget selv og deres motstander – rullet over 5 og 10 kamper. 
        Disse verdiene brukes som input til en regresjonsmodell som predikerer forventet antall mål for hvert lag.
        
        Når vi har estimert forventede mål $\\lambda_{home}$ og $\\lambda_{away}$, brukes en Poisson-fordeling til å beregne sannsynligheten for ulike kampresultater.
        Dette lar oss estimere sjanser for hjemmeseier, uavgjort og borteseier – og dermed også beregne "fair odds" for ulike spilltyper.
        """
    )

    st.subheader("Disclaimer")
    st.info(
        "Modellen tar kun hensyn til statistiske måldata og fanger ikke opp faktorer som skader, suspensjoner, form, taktiske endringer eller andre eksterne variabler."
    )
    st.warning(
        "Denne appen er kun ment for informasjons- og utdanningsformål og leveres uten garantier."
    )

    st.subheader("Feature-vekter per liga")
    league = st.selectbox(
        "Velg liga for å se modellens vekter",
        list(LEAGUES.keys()),
        key="model_info_league",
    )
    # Last inn modell for valgt liga
    model, _ = load_models_for_league(league, models_dir=f"{DATA_PATH}/models")

    stat_windows = {"xg": [5, 10], "gf": [5, 10], "ga": [5, 10]}
    feature_names = (
        [f"xg_roll{w}" for w in stat_windows["xg"]]
        + [f"gf_roll{w}" for w in stat_windows["gf"]]
        + [f"xg_conceded_away_roll{w}" for w in stat_windows["xg"]]
        + [f"ga_away_roll{w}" for w in stat_windows["ga"]]
        + ["avg_goals_for", "avg_goals_against_away"]
        + ["is_home"]
    )

    # Vis feature-vekter
    show_feature_weights(model, feature_names)
