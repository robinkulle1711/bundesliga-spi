# Bundesliga SPI Prediction Model - Step-by-Step Guide

This guide will take you from setup to a deployed Bundesliga prediction model, replicating FiveThirtyEight's Soccer Power Index (SPI) methodology.

## Project Overview

**Goal:** Build a prediction model for Bundesliga matches that:
- Calculates team ratings using Elo-style system
- Predicts match outcomes with probabilities
- Displays results in an interactive dashboard
- Serves as portfolio piece for future Mittelstand clients

**Reference:** [FiveThirtyEight SPI Methodology](https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/)

---

## Phase 1: Setup & Research (Week 1)

### Step 1: Deep Dive into FiveThirtyEight Methodology

Read and document the core components:
- SPI ratings (offensive and defensive strength)
- Elo-style updating after each match
- Expected goals framework
- Home field advantage adjustment
- Match outcome probabilities

**Action:** Create a `docs/methodology_notes.md` with key formulas and logic

### Step 2: Claude Code Setup

You've already done this! ✅

### Step 3: Assess Data Requirements

For an SPI-style model, you need:
- Historical match results (date, teams, score, home/away) - 2-5 years
- Current season fixtures
- Optional: xG (expected goals) data

**Data sources for Bundesliga:**
- **football-data.co.uk** - free historical results CSV files
- **FBref.com** - has xG data (can be scraped carefully)
- **api-football.com** - free tier for fixtures and basic stats
- **Transfermarkt** - supplementary team info

### Step 4: Set Up Project Structure

Run this Claude Code command:

```bash
claude-code "Create a Python project structure for a Bundesliga prediction model with folders: data/raw, data/processed, data/predictions, src, notebooks, docs, tests. Also create a requirements.txt with pandas, numpy, matplotlib, seaborn, streamlit, requests, scipy, and a basic README.md explaining this is a Bundesliga SPI-style prediction model. Include a .gitignore for Python projects."
```

**Expected structure:**
```
bundesliga-spi/
├── data/
│   ├── raw/           # Downloaded data
│   ├── processed/     # Cleaned data
│   └── predictions/   # Model outputs
├── src/
│   ├── data_prep.py
│   ├── elo_model.py
│   ├── calculate_ratings.py
│   ├── predict_matches.py
│   └── dashboard.py
├── notebooks/         # Jupyter notebooks for exploration
├── docs/             # Documentation
├── tests/            # Unit tests (optional)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Phase 2: Data Collection & Prep (Week 1-2)

### Step 5: Collect Historical Bundesliga Data

**Manual download:**
1. Go to https://www.football-data.co.uk/germanm.php
2. Download last 5 seasons of Bundesliga (D1.csv files)
3. Save them in `data/raw/` with names like `D1_2324.csv`, `D1_2223.csv`, etc.

**Or use Claude Code:**

```bash
claude-code "Write a Python script called src/download_data.py that downloads the last 5 seasons of Bundesliga data from football-data.co.uk and saves them to data/raw/. The URL pattern is https://www.football-data.co.uk/mmz4281/YYZZ/D1.csv where YYZZ is like 2324 for 2023-24 season. Handle the different season formats properly."
```

### Step 6: Explore the Data

```bash
claude-code "Create a Jupyter notebook called notebooks/01_data_exploration.ipynb that loads all the Bundesliga CSV files from data/raw/, combines them, and shows me: 1) what columns are available, 2) date range, 3) number of matches, 4) all unique team names, 5) basic statistics like average goals, home win percentage, 6) sample of the data."
```

**Run the notebook:**
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Step 7: Data Cleaning and Standardization

```bash
claude-code "Create src/data_prep.py that loads all Bundesliga CSV files from data/raw/, standardizes team names (handle variations like Bayern Munich vs Bayern München vs FC Bayern), creates a clean dataframe with columns: date, home_team, away_team, home_goals, away_goals, season. Save to data/processed/matches_clean.csv. Include a function to handle team name standardization with a mapping dictionary. Add error handling for missing data."
```

**Test it:**
```bash
python src/data_prep.py
```

---

## Phase 3: Build the Model (Week 2-3)

### Step 8: Implement Elo Rating System

```bash
claude-code "Create src/elo_model.py with a class called EloRating that:
1. Initializes all teams at 1500 rating
2. Has a method update_rating() that updates ratings after a match based on result
3. Uses K-factor of 32 (can be tuned later)
4. Accounts for home advantage (+100 rating points for home team in expectation calculation)
5. Has a method predict_outcome() that calculates win/draw/loss probabilities given two team ratings
6. Include detailed docstrings explaining the formulas (expected score, rating update)
7. Add a method to calculate expected goals based on ratings
Make it clean, well-documented code that I can show to clients."
```

**Key Elo formula to implement:**
- Expected score: E = 1 / (1 + 10^((opponent_rating - own_rating) / 400))
- Rating update: New_rating = Old_rating + K * (Actual_score - Expected_score)
- Where Actual_score is 1 for win, 0.5 for draw, 0 for loss

### Step 9: Calculate Historical Ratings

```bash
claude-code "Create src/calculate_ratings.py that:
1. Loads the clean match data from data/processed/matches_clean.csv
2. Uses the EloRating class to process all historical matches in chronological order
3. Tracks rating after each match for each team
4. Saves a CSV with columns: date, team, rating, match_id to data/processed/team_ratings_history.csv
5. Saves current ratings to data/processed/current_ratings.csv
6. Prints current top 10 teams by rating
7. Add visualization of rating evolution for top teams
Include progress updates while processing and handle edge cases like new teams."
```

**Run it:**
```bash
python src/calculate_ratings.py
```

### Step 10: Build Prediction Engine

```bash
claude-code "Create src/predict_matches.py that:
1. Loads current Elo ratings
2. Loads upcoming fixtures (we'll create this file manually for now)
3. For each match, calculates:
   - Win/draw/loss probabilities
   - Expected score for each team
   - Most likely outcome
4. Saves predictions to data/predictions/upcoming_predictions.csv with columns: date, home_team, away_team, home_win_prob, draw_prob, away_win_prob, predicted_home_goals, predicted_away_goals
5. Prints a formatted table of predictions
Make output easy to read and professional-looking."
```

### Step 11: Validate and Tune (Backtesting)

```bash
claude-code "Create notebooks/02_model_validation.ipynb that:
1. Loads our calculated Elo ratings history
2. For the 2024-25 season (or most recent complete season), simulates predictions:
   - For each match, use ratings BEFORE that match to predict outcome
   - Compare prediction to actual result
3. Calculates accuracy metrics:
   - Percentage of correct outcome predictions
   - Brier score for probabilistic accuracy
   - Log loss
   - Calibration plot (are 70% predictions right 70% of the time?)
4. Compares to naive baselines:
   - Always predict home win
   - Predict based on league position
5. Shows most confident correct and incorrect predictions
6. Visualizes results with charts
Include commentary on what the metrics mean."
```

**Run and analyze:**
```bash
jupyter notebook notebooks/02_model_validation.ipynb
```

Use results to tune K-factor, home advantage, etc.

---

## Phase 4: Build Dashboard (Week 3)

### Step 12: Create Upcoming Fixtures File

For now, create manually or use this command:

```bash
claude-code "Create src/get_fixtures.py that creates a sample upcoming fixtures file for the next Bundesliga matchday. Include columns: date, home_team, away_team, matchday. Save to data/raw/upcoming_fixtures.csv. Use real team names from our dataset. Later we can connect to an API for automatic updates."
```

### Step 13: Build Streamlit Dashboard

```bash
claude-code "Create src/dashboard.py using Streamlit with these sections:

1. HEADER
   - Title: 'Bundesliga Prediction Model (SPI-Style)'
   - Brief explanation of what the model does
   - Link to methodology documentation

2. CURRENT TEAM RATINGS
   - Table showing all teams with current rating, sorted by rating
   - Color-code by rating (green=high, yellow=medium, red=low)
   - Show offensive and defensive ratings if available

3. UPCOMING MATCH PREDICTIONS
   - For each upcoming match show:
     - Teams, date
     - Win/draw/loss probabilities as horizontal bars
     - Predicted score
     - Confidence level
   - Sort by date

4. RATING EVOLUTION
   - Line chart showing rating over time for selected teams
   - Dropdown to select which teams to display
   - Default to top 5 teams

5. MODEL PERFORMANCE
   - Display accuracy metrics from backtesting
   - Show calibration plot
   - Table of most confident predictions (right and wrong)

6. SIDEBAR
   - Model parameters (K-factor, home advantage)
   - Data last updated date
   - Link to GitHub repo

Make it clean, professional, and mobile-friendly. Use st.metric() for key numbers, st.plotly_chart() or st.line_chart() for visualizations."
```

### Step 14: Test Dashboard Locally

```bash
streamlit run src/dashboard.py
```

Open in browser, test all features, make sure it looks good.

### Step 15: Refine Dashboard

Iterate with Claude Code:

```bash
claude-code "Update dashboard.py to add a 'Team Deep Dive' tab where users can:
1. Select a specific team from dropdown
2. See their rating history as a line chart
3. See their last 5 matches with actual results vs predictions
4. See their next 3 fixtures with predictions
5. Show their offensive vs defensive rating on a scatter plot compared to league average"
```

```bash
claude-code "Add a 'Surprising Results' section showing matches where:
1. Predicted probability was <20% but outcome happened
2. Display with actual result, prediction, and rating impact
3. Limit to last 10 most surprising results"
```

---

## Phase 5: Deploy & Document (Week 4)

### Step 16: Prepare for Deployment

```bash
claude-code "Create .streamlit/config.toml with a professional theme (use dark or light theme, nice colors). Also verify requirements.txt includes all dependencies for the dashboard: streamlit, pandas, numpy, plotly, matplotlib. Add any missing ones."
```

### Step 17: Create Comprehensive README

```bash
claude-code "Update README.md to be a professional portfolio-quality document with:

1. PROJECT OVERVIEW
   - What this is and why I built it
   - Live demo link (placeholder for now)
   - Screenshot of dashboard

2. FEATURES
   - List key features with emojis
   - What makes this different/special

3. METHODOLOGY
   - Brief explanation of Elo rating system
   - Link to FiveThirtyEight methodology
   - Our adaptations and parameters

4. INSTALLATION
   - Requirements
   - Step-by-step setup instructions
   - How to run locally

5. USAGE
   - How to update with new results
   - How to generate predictions
   - How to run dashboard

6. MODEL PERFORMANCE
   - Accuracy metrics
   - Comparison to baselines
   - Limitations and caveats

7. FUTURE IMPROVEMENTS
   - Planned enhancements
   - Ideas for expansion

8. TECHNICAL DETAILS
   - Tech stack
   - Data sources
   - File structure

9. CREDITS
   - Reference to FiveThirtyEight
   - Data sources
   - Libraries used

Make it professional, clear, and impressive for a portfolio piece."
```

### Step 18: Create Methodology Documentation

```bash
claude-code "Create docs/methodology.md with detailed explanation:

1. ELO RATING SYSTEM BASICS
   - History and origin
   - How it works conceptually
   - Why it's good for soccer

2. OUR IMPLEMENTATION
   - Initial ratings
   - K-factor choice and reasoning
   - Home advantage factor
   - Update formula with examples

3. CONVERTING RATINGS TO PREDICTIONS
   - Probability calculation
   - Expected goals estimation
   - Draw probability handling

4. MODEL VALIDATION
   - How we tested it
   - Metrics we use and why
   - Results and interpretation

5. LIMITATIONS
   - What the model doesn't account for (injuries, tactics, etc.)
   - When predictions are less reliable
   - Known biases

6. COMPARISON TO FIVETHIRTYEIGHT
   - What we adopted
   - What we simplified
   - Potential improvements

Write this for both technical and non-technical audiences - explain concepts clearly but include formulas for those who want detail."
```

### Step 19: Deploy to Streamlit Cloud

1. **Commit everything to Git:**
```bash
git add .
git commit -m "Complete Bundesliga prediction model with dashboard"
```

2. **Create GitHub repository:**
   - Go to GitHub.com
   - Create new repository: `bundesliga-spi-model`
   - Don't initialize with README (you have one)

3. **Push to GitHub:**
```bash
git remote add origin https://github.com/YOUR_USERNAME/bundesliga-spi-model.git
git branch -M main
git push -u origin main
```

4. **Deploy on Streamlit Cloud:**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `src/dashboard.py`
   - Click "Deploy"
   - Wait 2-3 minutes
   - You'll get a public URL like `your-app.streamlit.app`

5. **Update README with live link**

### Step 20: Write Blog Post / LinkedIn Article

```bash
claude-code "Help me create an outline for a LinkedIn article (in German and English versions) titled:

German: 'Bundesliga-Vorhersagen mit KI: Ein Daten-Projekt'
English: 'Building a Bundesliga Prediction Model: A Data Science Project'

Structure:
1. Hook - why this project was interesting
2. The challenge - what FiveThirtyEight does
3. My approach - tools and methodology
4. Key learnings - technical and practical
5. Interesting findings - surprising results from the model
6. Next steps - how this applies to business analytics
7. Call to action - link to live dashboard, GitHub

Make it engaging but professional. Include 2-3 data visualizations from my dashboard. Keep it under 1000 words. End with a bridge to business applications - 'The same techniques used here for predicting soccer matches can forecast sales, optimize inventory, or predict equipment failures in manufacturing.'"
```

Write this in a Google Doc or Notion, get feedback, polish it.

---

## Phase 6: Launch & Iterate (Week 4-5)

### Step 21: Strategic Sharing

**LinkedIn post (German + English):**
- Announce your project
- Include screenshot of dashboard
- Link to live demo and GitHub
- Tag relevant people/pages (data science groups, soccer analytics)
- Use hashtags: #DataScience #Bundesliga #Python #MachineLearning

**Share in communities:**
- r/Bundesliga (check rules, don't spam)
- r/soccer (Soccer Saturday or Free Talk Friday threads)
- German soccer forums (transfermarkt.de, etc.)
- Data science communities
- Tweet/X with visualizations

**Local outreach:**
- Email Werder Bremen fan sites (Werderfreunde.de, Deichstube)
- Contact sports section of Weser-Kurier
- Share in Bremen tech/data communities

### Step 22: Weekly Updates

Once live, update weekly:
- Run predictions for upcoming matchday
- Post "Model says Bayern 73% to beat Dortmund" type content
- Track accuracy, share results
- Build following by being consistent

### Step 23: Gather Feedback and Iterate

- Monitor comments and questions
- Track which predictions were right/wrong
- Refine model based on performance
- Add features people request

---

## Phase 7: Pivot to Mittelstand (Ongoing from Week 4)

### Step 24: Extract Business Lessons

Create `docs/business_case_study.md`:

**"From Soccer Predictions to Business Forecasting"**

Parallels to draw:
- Historical match data → Historical sales/production data
- Team ratings → Customer segments, product performance
- Match predictions → Sales forecasts, demand prediction
- Probability outputs → Risk assessment, confidence intervals
- Model validation → Forecast accuracy tracking

**Key message:** "The same statistical techniques that predict Bundesliga matches can forecast your business outcomes."

### Step 25: Update Your Professional Presence

**LinkedIn profile:**
- Headline: "Data Analyst | Building Predictive Models | Helping Manufacturing Companies Make Data-Driven Decisions"
- Featured section: Pin your Bundesliga project
- About section: Mention both your corporate experience AND side projects

**Create simple landing page:**
- yourname.github.io or simple Carrd.co site
- Showcase: Soccer project + "I can do this for your business data"
- Services: Data pipeline automation, forecasting, dashboard development
- Contact form

### Step 26: Identify Mittelstand Prospects

Research 10-20 companies in Bremen/Niedersachsen:

**Target profile:**
- Manufacturing companies
- 50-500 employees
- Growing (look for job postings, expansion news)
- Multiple locations or complex operations
- Industries: automotive, maritime, food/beverage, industrial equipment

**Research on:**
- LinkedIn (find decision makers)
- Company websites (look for pain points)
- News articles (growth, challenges)
- Glassdoor reviews (internal process issues)

**Create spreadsheet:**
- Company name, industry, size
- Decision maker name and title
- LinkedIn profile
- Potential pain points
- Contact info
- Notes

### Step 27: Craft Outreach Strategy

**Email template (in German):**

```
Betreff: Predictive Analytics für [Company Name]

Guten Tag [Name],

ich habe kürzlich ein Vorhersagemodell für Bundesliga-Spiele entwickelt [Link], 
das historische Daten nutzt, um zukünftige Ergebnisse vorherzusagen.

Derselbe Ansatz – Vergangenheitsdaten zur Prognose nutzen – kann 
Produktionsunternehmen wie [Company] helfen bei:
- Präziserer Absatzprognose
- Optimierter Produktionsplanung  
- Besserer Lagerbestandsführung

Ich arbeite als Data Analyst in der Fertigung und bin in Bremen ansässig. 
Gerne zeige ich Ihnen, was mit Ihren Daten möglich ist.

Interesse an einem 15-minütigen Gespräch?

Beste Grüße,
[Your name]

P.S. Hier ist mein Bundesliga-Projekt: [dashboard link]
```

**Alternative: LinkedIn message:**
Keep it shorter, more conversational, reference their company specifically.

### Step 28: Start Outreach (Gradually)

Don't spam everyone at once:
- Week 1: Reach out to 3 companies (test messaging)
- Get responses, refine approach
- Week 2: Reach out to 5 more
- Aim for 1-2 discovery calls

**In discovery calls:**
- Ask about their current data challenges
- Don't pitch immediately - understand their needs
- Show your soccer project as proof of capability
- Offer small pilot project (1-2k€) to prove value

---

## Success Metrics

### Technical Success:
- ✅ Model achieves >55% accuracy on match outcomes
- ✅ Brier score better than naive baseline
- ✅ Dashboard deployed and accessible
- ✅ Code is clean, documented, professional

### Portfolio Success:
- ✅ Shared publicly, got engagement (likes, comments, shares)
- ✅ GitHub repo has clear README and documentation
- ✅ Can confidently explain how it works to non-technical people
- ✅ Have learned 3+ new technical skills

### Business Pivot Success:
- ✅ List of 10+ qualified Mittelstand prospects
- ✅ Outreach materials prepared and tested
- ✅ Had 2-3 discovery calls with potential clients
- ✅ Portfolio clearly demonstrates business value

---

## Troubleshooting Common Issues

### Data Issues:
**Problem:** Team names don't match across seasons
**Solution:** Create comprehensive mapping dictionary in `data_prep.py`

**Problem:** Missing data for some matches
**Solution:** Add validation, log issues, decide how to handle (skip or estimate)

### Model Issues:
**Problem:** Predictions seem off / not accurate enough
**Solution:** 
- Check K-factor (try 20-40 range)
- Adjust home advantage (try 50-150 range)
- Validate data quality
- Check if using ratings from BEFORE match in backtest

**Problem:** Draw predictions too low/high
**Solution:** Adjust draw probability calculation - might need separate parameter

### Dashboard Issues:
**Problem:** Dashboard loads slowly
**Solution:** Cache data loading with `@st.cache_data` decorator

**Problem:** Charts not displaying
**Solution:** Check data types, ensure dates are parsed correctly, verify plotly is installed

### Deployment Issues:
**Problem:** App won't deploy on Streamlit Cloud
**Solution:** Check requirements.txt, ensure all imports are listed, check Python version compatibility

**Problem:** Data files not found after deployment
**Solution:** Use relative paths, ensure data files are in repo (check .gitignore)

---

## Next-Level Enhancements (Future)

Once basic model is working:

1. **Offensive/Defensive Split:** 
   - Separate ratings for attacking and defending strength
   - More nuanced predictions

2. **Expected Goals (xG) Integration:**
   - Scrape xG data from FBref
   - Use to improve goal predictions

3. **2. Bundesliga Coverage:**
   - Apply model to second division
   - Werder Bremen context!

4. **League Table Projections:**
   - Simulate rest of season
   - Predict final standings with probabilities

5. **Player-Level Analysis:**
   - Track individual player impact
   - Injury/suspension adjustments

6. **Multi-League Expansion:**
   - Premier League, La Liga, etc.
   - Cross-league comparisons

7. **Live Updates:**
   - Automatic data refresh
   - Real-time rating updates during matches

8. **Betting Value Finder:**
   - Compare your probabilities to betting odds
   - Identify value bets
   - Track performance

---

## Resources

### Learning Resources:
- FiveThirtyEight SPI Methodology: https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/
- Elo Rating System: https://en.wikipedia.org/wiki/Elo_rating_system
- Expected Goals (xG): https://fbref.com/en/expected-goals-model-explained/
- Streamlit Documentation: https://docs.streamlit.io/

### Data Sources:
- football-data.co.uk: Historical match results
- FBref.com: Advanced statistics, xG data
- api-football.com: Live fixtures and data
- Transfermarkt: Team/player information

### Python Libraries:
- pandas: Data manipulation
- numpy: Numerical computations
- streamlit: Dashboard framework
- plotly: Interactive visualizations
- scipy: Statistical functions

---

## Contact & Feedback

As you build this, document:
- What worked well
- What was harder than expected
- Interesting findings
- Questions that came up

This becomes content for your blog post and helps you explain the project to clients!

---

## Final Checklist Before Launch

- [ ] All code is clean and documented
- [ ] README is complete and professional
- [ ] Dashboard works on mobile
- [ ] GitHub repo is public
- [ ] Live demo URL works
- [ ] Blog post is written and reviewed
- [ ] LinkedIn post is drafted
- [ ] Shared in 3+ communities
- [ ] Business case study is documented
- [ ] 10 Mittelstand prospects identified
- [ ] Outreach email templates tested

---

**You've got this! Start with Step 4 (project structure) and work your way through. Use Claude Code for every step - that's what it's built for.**

**When you get stuck, ask Claude Code for help. When you finish a section, commit to Git. Build incrementally, test frequently, and ship it!**

Good luck! 🚀⚽📊
