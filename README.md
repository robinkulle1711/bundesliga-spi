# Bundesliga Prediction Model (SPI-Style)

A FiveThirtyEight-style Soccer Power Index (SPI) prediction model for the Bundesliga, built with Python and Streamlit.

## What this does

- Calculates team strength ratings using an Elo-based system trained on 5 seasons of historical Bundesliga data
- Predicts match outcomes (win/draw/loss probabilities and expected goals)
- Displays results in an interactive Streamlit dashboard

## Project structure

```
├── data/
│   ├── raw/          # Downloaded CSV files from football-data.co.uk
│   ├── processed/    # Cleaned and standardized data
│   └── predictions/  # Model output files
├── src/
│   ├── download_data.py      # Fetch historical data
│   ├── data_prep.py          # Clean and standardize
│   ├── elo_model.py          # Elo rating system
│   ├── calculate_ratings.py  # Run ratings over history
│   ├── predict_matches.py    # Generate predictions
│   ├── get_fixtures.py       # Fetch upcoming fixtures
│   └── dashboard.py          # Streamlit app
├── notebooks/        # Exploration and validation notebooks
├── docs/             # Methodology and documentation
├── tests/            # Unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download historical data
python src/download_data.py

# 2. Clean and prepare data
python src/data_prep.py

# 3. Calculate ratings
python src/calculate_ratings.py

# 4. Generate predictions
python src/predict_matches.py

# 5. Launch dashboard
streamlit run src/dashboard.py
```

## Data source

Historical match results from [football-data.co.uk](https://www.football-data.co.uk/germanm.php).

## Methodology

See [docs/methodology_notes.md](docs/methodology_notes.md) for details on the Elo rating system and prediction formulas.

## Tech stack

Python · pandas · numpy · scipy · Streamlit · Plotly
