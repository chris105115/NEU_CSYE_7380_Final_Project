# Stock Trading Dashboard with Wyckoff Analysis

This application provides a stock trading dashboard with backtesting capabilities and Wyckoff analysis insights. It features a chatbot for investment insights and a backtesting dashboard for trading strategy analysis.

## Features

- Interactive stock trading dashboard
- Wyckoff analysis chatbot
- Backtesting capabilities with Q-learning
- Real-time stock data visualization
- Portfolio performance tracking

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your virtual environment is activated.

2. Start the Streamlit application:
```bash
streamlit run app.py
```

3. The application will open in your default web browser. If it doesn't, you can access it at:
```
http://localhost:8501
```

## Usage

### Chatbot Mode
- Select "Chatbot" from the sidebar
- Enter your questions about investment insights
- Get responses based on Wyckoff's methodology

### Backtesting Dashboard
- Select "Back-testing Dashboard" from the sidebar
- Enter a stock ticker symbol (e.g., "NVDA")
- Select start and end dates
- Set the number of training episodes
- Click "Run Back-test" to analyze the trading strategy

## Project Structure

- `app.py` - Main application file
- `requirements.txt` - Python package dependencies

## Dependencies

- streamlit==1.32.0
- yfinance==0.2.36
- pandas==2.2.0
- numpy==1.26.4
- matplotlib==3.8.3

## Notes

- The application uses Yahoo Finance API for stock data
- Backtesting results are for educational purposes only
- Always do your own research before making investment decisions

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check if your virtual environment is activated
3. Ensure you have a stable internet connection for stock data
4. Verify that the stock ticker symbol is valid

## License

This project is licensed under the MIT License - see the LICENSE file for details. 