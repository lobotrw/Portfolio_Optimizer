### Portfolio Optimization Application

**This Streamlit-based web application allows users to optimize their investment portfolios using Modern Portfolio Theory (MPT). Here's a breakdown of its key functionalities and features:**

![plot](images/mpt.png)

#### Features:

2. **File Upload**:
   - Users can upload a CSV file containing asset price data. The application expects the file to have a date column (`time`) in `MM-DD-YYYY` format and asset price columns in numerical format.
   - Displays an example CSV format to guide users.
![plot](images/upload.png)


4. **Portfolio Optimization Methods**:
   - **Sharpe Ratio Optimization**: Maximizes the Sharpe ratio to optimize the portfolio.
   - **Min Volatility Optimization**: Minimizes the portfolio volatility.

5. **Interactive User Inputs**:
   - Users can select the optimization method (Sharpe or Min Volatility) through a radio button.
   - Adjust the range of weights for the assets using a slider.
![plot](images/settings.png)

6. **Optimization Execution**:
   - Once the user sets the parameters and clicks the "Run optimization" button, the application computes the optimal weights using the selected method.
   - Displays the optimal weights as both a bar chart and a DataFrame.
![plot](images/weights.png)

7. **Portfolio Performance Metrics**:
   - Calculates various performance metrics for the optimized portfolio, including:
     - Cumulative returns
     - Sharpe ratio
     - Sortino ratio
     - Omega ratio
     - Average daily, monthly, and annual volatility
     - Maximum drawdown
![plot](images/perf.png)


This application provides a comprehensive and user-friendly interface for optimizing investment portfolios, making it a valuable tool for both novice and experienced investors.

## Method notes (logic fixes)

- **Portfolio value is weighted by capital.** Each asset gets its weight of the starting capital: `sum_i w_i * P_i(t)/P_i(0)`, buy and hold. The old code summed `w_i * P_i(t)` (price levels), so the expensive asset dominated. For example, a 65% BTC weight became about 98% of the portfolio.
- **Out-of-sample by default.** Weights are optimised on the first 70% of the rows (you can change this with a slider), and performance is reported on the remaining 30%, next to an equal-weight benchmark and the in-sample numbers. At 100% the result is explicitly labelled *in-sample (hindsight)*.
- **Annualisation** uses 365 periods per year when the data contains weekend dates (crypto), and 252 otherwise. This applies to PyPortfolioOpt and to every metric. Monthly volatility uses `sqrt(freq/12)`.
- **Total return** is shown as `(growth − 1) × 100`. The old "Cumulative returns" showed growth × 100. The chart shows growth of $1.
- **Ratios:** there is an optional risk-free rate. Sortino uses the standard downside deviation. If max-Sharpe is infeasible (no asset beats the risk-free rate), the app falls back to min-volatility and says so.
- The random portfolios on the frontier plot respect the weight bounds. Dates in the `time` column can also be in other unambiguous formats.
