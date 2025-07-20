# ⚽ Football Match Outcome Prediction

This project provides a machine learning-powered prediction system for football matches, including:

- Match outcome probabilities (home win, draw, away win)
- Fair odds estimation
- Match grouping and interactive display
- Support for multiple leagues (e.g., Premier League)

It uses data collected from the web, feature engineering, and Poisson regression models trained on historical data.

## 🔗 Live App

Try the deployed app here:  
👉 [https://sindreulsaker-football-prediction-app.streamlit.app](https://sindreulsaker-football-prediction-app.streamlit.app)

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
│   │   └── odds.py             # Calculate different odds using poisson models
│   ├── scripts/
│   │   └── update_all.py       # Pipeline runner: fetch → process → train
│   ├── ui_components/
│   │   └── display.py          # Display logic for prediction results
│   └── ui_pages/
│       ├── main.py             # Streamlit main page and navigation
│       ├── predictions.py      # Predictions view
│       ├── oddschecker.py      # Odds comparison tool
│       └── model_info.py       # Info and disclaimers about the model

---

## 🚀 Features

- 📊 **Poisson-based match prediction** (expected goals → outcome probabilities)
- 🔁 **Supports multiple leagues**
- 🧠 **Custom feature engineering**
- 🧼 **Automated fetch-process-train pipeline**
- 🖥 **Streamlit-based UI** with expandable daily match view
- 🔀 **Toggle between probabilities and fair odds**

---

## 🧪 Installation (for local use)

1. Clone this repository:

   ```bash
   git clone https://github.com/SindreUlsaker/Football_prediction.git
   cd Football_prediction
   
2. Install dependencies:

   ```bash
   pip install -r requirements.txt

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
