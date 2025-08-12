# ⚽ Football Match Outcome Prediction

This project provides a machine learning-powered prediction and simulation system for football matches, including:

- Match outcome probabilities (home win, draw, away win) using Poisson regression
- Season simulation for league tables and long-term forecasts
- Fair odds estimation for multiple bet types (1X2, BTTS, Over/Under)
- Match grouping, filtering, and interactive display
- Support for multiple leagues (e.g., Premier League, La Liga, Serie A, Bundesliga, Ligue 1)

It uses data collected from the web, feature engineering, and Poisson regression models trained on historical data.

## 🔗 Live App

Try the deployed app here:  
👉 [https://football-predictions.streamlit.app](https://football-predictions.streamlit.app)

---

## 🏗 Project Structure

```text
football_prediction/
│
├── app.py                      # Entry point for the Streamlit app
├── requirements.txt            # Python dependencies
├── config/
│   ├── leagues.py              # League-specific settings
│   └── settings.py             # Centralized base path configuration for project-wide file access
│
├── data/
│   ├── raw/                    # Raw scraped data
│   ├── processed/              # Processed and feature-engineered datasets
|   |    └──simulations/        # Saved simulation for each league
│   └── models/                 # Saved model and scaler files (.joblib)
│
├── src/
│   ├── data/
│   │   ├── fetch.py            # Web scraping functions (e.g., from fbref.com)
│   │   └── process.py          # Data cleaning and processing
│   ├── features/
│   │   └── features.py         # Feature engineering
│   ├── models/
│   │   ├── train.py            # Model training
│   │   ├── predict.py          # Model loading and prediction
│   │   ├── odds.py             # Calculate different odds using poisson models
│   │   └── simulate.py         # Simulate the rest of the games for a given league
│   ├── scripts/
│   │   ├── update_all.py       # Pipeline runner: fetch → process → train
│   │   ├── fetch_prev_season   # Used in update_all_annual to fetch previous season
│   │   ├── daily_merge.py      # Merge previous season data with current season data
│   │   └── simulate_all.py     # Simulate the rest of the games for all leagues
│   ├── ui_components/
│   │   └── display.py          # Display logic for prediction results
│   └── ui_pages/
│       ├── main.py             # Streamlit main page and navigation
│       ├── predictions.py      # Predictions view
│       ├── oddschecker.py      # Odds comparison tool
│       ├── simulator.py        # Show simulated outcomes for all leagues
│       └── model_info.py       # Info and disclaimers about the model
│
├── test/                       # Test files for all core files

```

---

## 🚀 Features

- 📊 **Poisson-based match prediction** (expected goals → outcome probabilities)
- 🎯 **Match simulation** for season-long projections
- 🔁 **Automated daily fetch–process–train pipeline**
- 🖥 **Streamlit-based UI** with league filters and date-based match view
- 🎲 **Odds tools** for 1X2, BTTS, and Over/Under fair odds calculation

---

## 🧪 Installation (for local use)

1. Clone this repository:

   ```bash
   git clone https://github.com/SindreUlsaker/Football_prediction.git
   cd Football_prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 📝 Requirements

The full list of Python packages used is in `requirements.txt`. Key ones include:

- `streamlit`
- `pandas`
- `scikit-learn`
- `joblib`
- `selenium`
- `beautifulsoup4`
- `lxml`

---

## 🛠 Author & Contact

Developed by [Sindre Ulsaker](https://github.com/SindreUlsaker)  
Project created for educational and demonstration purposes.

---

## 📄 License

This project is open source under the MIT License.
