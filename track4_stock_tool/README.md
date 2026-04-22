# ACC102 Track4: Professional Stock Analytics Dashboard
Xi'an Jiaotong-Liverpool University | ACC102 Mini Assignment | Track 4
Student Name: Yangyang.Yan | Student ID: 2472893

## Project Overview
This is an interactive financial analysis application built with Streamlit. It connects to the WRDS CRSP database to obtain official stock data, performs quantitative risk and return analysis, generates interactive visualizations, and provides investment analysis based on the CFA framework.

## Core Functions
- WRDS database connection and stock data extraction
- Calculation of financial indicators: moving averages, RSI, Bollinger Bands, Sharpe Ratio, volatility, maximum drawdown
- Interactive charts: price trends, trading volume, RSI, cumulative return comparison
- Intelligent technical analysis and CFA investment recommendations
- Data export (CSV/Excel format)

## Data Source
- Database: WRDS CRSP
- Data Table: crsp.dsf (Daily Stock File)
- Usage: For academic and educational purposes only

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Launch the application: streamlit run app.py
3. Log in with your WRDS account in the sidebar
4. Enter stock ticker, date range and view analysis

## File Structure
- app.py: Main program code
- requirements.txt: Project dependency packages
- README.md: Project documentation

## Disclaimer
This project is for academic learning only and does not constitute investment advice. All investments involve risks.

## AI Use Disclosure
AI Tool: Doubao and Gemini Pro
Usage: Assisted with UI design and code optimization. Core financial logic and data processing are independently completed.