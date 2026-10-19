# Imports
import time

import numpy as np
import pandas as pd
import streamlit as st
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from streamlit.components.v1 import html

from functions import portfolio_optimization as po

# ------------------------------------------------------------------------------------------------------------------------------------
#Set Page Configurations
# ------------------------------------------------------------------------------------------------------------------------------------

st.set_page_config(
                    page_title = "Portfolio Optimization",
                    page_icon  = "📊",
                    layout     = "wide"
                                )

# ------------------------------------------------------------------------------------------------------------------------------------
# Upload a file
# ------------------------------------------------------------------------------------------------------------------------------------
c1_,c2_,c3_= st.columns([1,1,1])

c_1,c_2,c_3= st.columns([1,6,1])


c2_.title("**Portfolio :blue[Optimization]**")

with st.sidebar:
    st.title("**Portfolio :blue[Optimization]**")

uploaded_file = c_2.file_uploader(":blue[Upload your .csv file with assets prices]")
c_2.write("---")
c_2.info('''**Ensure your .csv file matches the provided example to avoid errors.**\n 
            ⚠️FORMAT: [time column: MM-DD-YYYY, Asset name columns: 0.000]''', icon="📝")

c_2.write("***Example .csv***")
c_2.image("images/image.png")


if uploaded_file is not None:
    progress_text = "Operation in progress. Please wait."
    my_bar = c_2.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()

    up_done = c_2.success('Uploaded', icon="✅")
    time.sleep(1.0)
    up_done.empty()

    df = pd.read_csv(uploaded_file)
    if 'time' not in df.columns:
        c_2.error("The CSV needs a 'time' column (see the example below).")
        st.stop()
    # Setting Date as index: the documented MM-DD-YYYY format first, then any unambiguous date format
    try:
        df['time'] = pd.to_datetime(df['time'], format='%m-%d-%Y')
    except (ValueError, TypeError):
        try:
            df['time'] = pd.to_datetime(df['time'], format='mixed')
        except (ValueError, TypeError) as err:
            c_2.error(f"Could not parse the 'time' column as dates: {err}")
            st.stop()
    df = df.set_index('time').sort_index()
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').ffill().dropna()

if uploaded_file is None:
    c_2.warning('Please upload a file.')
    st.stop()

# ------------------------------------------------------------------------------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------------------------------------------------------------------------------

c_2.write("###")
c_2.write("---")
c_2.write("###")

freq = po.infer_frequency(df.index)

c_2.subheader("Method of Optimization")
method = c_2.radio(
                  "Set a method and Run optimization ⚙️",
                  key        =    "visibility",
                  options    =   ["Sharpe", "Min Volatility", "Omega"],
                  horizontal =   True
                    )

values = c_2.slider(
                    "Select a range of weights %",
                    0.0, 1.0, (0.05, 0.65)
                    )

train_pct = c_2.slider(
    "Training share of the data (%)", 30, 100, 70, step=5,
    help="Weights are optimised on the first part of the data and evaluated on the rest (out-of-sample). "
         "100 % = optimise and evaluate on the same data (in-sample, hindsight-biased).")
rf = c_2.number_input("Risk-free rate (annual, %)", min_value=0.0, max_value=20.0, value=0.0, step=0.25) / 100

c_2.caption(f"Detected {freq} periods per year ({'weekend dates present: 24/7 market' if freq == 365 else 'no weekend dates: trading days'}); "
            "used for all annualisation.")

train, test = po.train_test_split(df, train_pct / 100)
if len(train) < 30 or (test is not None and len(test) < 30):
    c_2.error("Not enough rows for this split: both the training and the test part need at least 30 rows.")
    st.stop()

# Calculate expected returns and sample covariance on the TRAINING data only
mu = expected_returns.mean_historical_return(train, frequency=freq)
S  = risk_models.sample_cov(train, frequency=freq)

ef = EfficientFrontier(mu, S, weight_bounds=values)

button = c_2.button("Run optimization")
if not button:
    st.stop()

n_assets = len(df.columns)
if values[0] * n_assets > 1 or values[1] * n_assets < 1:
    c_2.error(f"Weight range {values[0]:.2f}-{values[1]:.2f} is infeasible for {n_assets} assets: "
              f"the weights must be able to sum to 1 (min x n <= 1 <= max x n).")
    st.stop()

c_2.write("---")
c_2.subheader(f"Optimal :blue[{method}] Weights")
c_2.caption(f"Optimised on {train.index[0]:%Y-%m-%d} – {train.index[-1]:%Y-%m-%d} ({len(train)} rows).")
c_2.write("###")

custom = None
if method == "Sharpe":
    try:
        ef.max_sharpe(risk_free_rate=rf)
    except Exception as err:   # e.g. no asset beats the risk-free rate, or solver failure
        c_2.warning(f"Max-Sharpe optimization failed ({err}). Falling back to **Min Volatility** instead.")
        ef = EfficientFrontier(mu, S, weight_bounds=values)
        method = "Min Volatility (fallback)"
        try:
            ef.min_volatility()
        except Exception as err2:
            c_2.error(f"Min-volatility optimization failed too: {err2}")
            st.stop()
    cleaned_weights = ef.clean_weights()
elif method == "Min Volatility":
    try:
        ef.min_volatility()
    except Exception as err:
        c_2.error(f"Min-volatility optimization failed: {err}")
        st.stop()
    cleaned_weights = ef.clean_weights()
else:  # Omega
    returns = train.pct_change().dropna()
    custom = po.optimize_omega(returns=returns, threshold=0.0, bounds=values)
    cleaned_weights = dict(zip(df.columns, np.round(custom, 5)))

c_2.bar_chart(cleaned_weights, width='stretch', height=500)
c_2.dataframe(cleaned_weights, width='stretch')
c_2.write("---")

with c_2:
    po.plot_ef(5000, ef, method, custom_weights=custom, bounds=values)

weights = pd.Series(cleaned_weights, dtype=float)

# ------------------------------------------------------------------------------------------------------------------------------------
# Portfolio Performance
# ------------------------------------------------------------------------------------------------------------------------------------

def metrics(prices, w):
    port = po.portfolio_value(prices, w)          # capital-weighted buy-and-hold, starts at 1.0
    return port, {
        "Total return":          f"{(port.iloc[-1] - 1) * 100:.2f} %",
        "Sharpe Ratio":          f"{po.sharpe(port, freq, rf):.2f}",
        "Sortino Ratio":         f"{po.sortino(port, freq, rf):.2f}",
        "Omega Ratio":           f"{po.omega(port):.2f}",
        "AVG Daily Volatility":  f"{po.daily_vol(port) * 100:.2f} %",
        "AVG Monthly Volatility": f"{po.daily_vol(port) * 100 * np.sqrt(freq / 12):.2f} %",
        "AVG Annual Volatility": f"{po.daily_vol(port) * 100 * np.sqrt(freq):.2f} %",
        "Max Draw Down":         f"{po.max_drawdown(port) * 100:.2f} %",
    }

equal = pd.Series(1 / n_assets, index=df.columns)
eval_prices = test if test is not None else train
label = "out-of-sample (test)" if test is not None else "IN-SAMPLE (hindsight: same data used to optimise)"

c_2.write("---")
c_2.subheader("Portfolio :blue[Performance]")
c_2.caption(f"Evaluated {label}: {eval_prices.index[0]:%Y-%m-%d} – {eval_prices.index[-1]:%Y-%m-%d}. "
            "Buy-and-hold, each asset gets its weight of the starting capital.")

port, m_opt = metrics(eval_prices, weights)
port_eq, m_eq = metrics(eval_prices, equal)
_, m_in = metrics(train, weights)

with c_2:
    st.line_chart(pd.DataFrame({"Optimised portfolio": port, "Equal weight": port_eq}),
                  y_label="Growth of $1")
    st.write("---")
    table = pd.DataFrame({f"Optimised ({'test' if test is not None else 'in-sample'})": m_opt,
                          "Equal weight (same period)": m_eq})
    if test is not None:
        table["Optimised (train, in-sample)"] = pd.Series(m_in)
    st.table(table)
    st.area_chart(po.rol_max_drawdown(port) * 100, color="#d14747", y_label="Drawdown %")

c1_,c2_,c3_= st.columns([0.5,6,0.5])

c2_.write("---") 
c2_.write("**About**")

c2_.write(
'''
Created by @VanHe1sing  
       
TradingView: https://www.tradingview.com/u/VanHe1sing/#published-scripts     
Telegram: https://t.me/IvanKocherzhat
X: https://x.com/sxJEoRg7wwLR6ug


*This site is for informational purposes only.*\n
*· The information on our website is not financial advice, and you should not consider it to be financial advice.*
'''
)

