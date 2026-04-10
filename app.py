# ============================
# IPL 2025 Batters Dashboard
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("IPL2025Batters.csv")

# Normalize column names to lowercase and fix spaces
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={"player name": "player_name"})

# ----------------------------
# 2. Data Cleaning
# ----------------------------
numeric_cols = ['runs','matches','inn','no','hs','avg','bf','sr','100s','50s','4s','6s']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

df['avg'] = df['avg'].fillna(df['avg'].mean())
df['hs'] = df['hs'].fillna(df['hs'].median())

df = df.dropna(subset=['runs','matches','inn','avg','sr']).reset_index(drop=True)

# ----------------------------
# 3. Feature Engineering
# ----------------------------
df['boundary_runs'] = df['4s']*4 + df['6s']*6
df['boundary_pct'] = (df['boundary_runs']/df['runs'])*100
df['consistency_score'] = df['avg'] * df['sr']

# ----------------------------
# 4. Multi-tab Dashboard
# ----------------------------
st.title("🏏 IPL 2025 Batters Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "👤 Player Analysis", "🏟️ Team Comparison", "🤖 ML Predictions"])

# ----------------------------
# Tab 1: EDA
# ----------------------------
with tab1:
    st.header("Exploratory Data Analysis")

    st.subheader("Top 10 Run Scorers")
    top_scorers = df.sort_values(by='runs', ascending=False).head(10)
    st.bar_chart(top_scorers.set_index("player_name")['runs'])

    st.subheader("Strike Rate Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['sr'], bins=20, kde=True, ax=ax, color="blue")
    ax.set_title("Distribution of Strike Rates")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Between Features")
    st.pyplot(fig)

# ----------------------------
# Tab 2: Player Analysis
# ----------------------------
with tab2:
    st.header("Player Analysis")
    player = st.selectbox("Select a Player", df['player_name'].unique())
    pdata = df[df['player_name'] == player].iloc[0]

    st.write("### Player Stats")
    st.write(pdata)

    st.subheader("Boundary Contribution")
    st.write(f"{player}'s boundary contribution: {pdata['boundary_pct']:.2f}%")

# ----------------------------
# Tab 3: Team Comparison
# ----------------------------
with tab3:
    st.header("Team Comparison")

    team_runs = df.groupby("team")['runs'].sum().sort_values(ascending=False)
    st.subheader("Total Runs by Team")
    st.bar_chart(team_runs)

    team_avg = df.groupby("team")['avg'].mean().sort_values(ascending=False)
    st.subheader("Average Batting Performance by Team")
    st.bar_chart(team_avg)

# ----------------------------
# Tab 4: ML Predictions
# ----------------------------
with tab4:
    st.header("ML Predictions")

    X = df[['matches','inn','bf','sr','avg','50s','100s']]
    y = df['runs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Model Performance")
    st.write("R² Score:", r2_score(y_test, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    coeffs = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    st.subheader("Feature Importance (Coefficients)")
    st.write(coeffs)