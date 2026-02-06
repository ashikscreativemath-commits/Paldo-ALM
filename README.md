# Paldo ALM (Adaptive Learning Machine) 🧠📈

**Paldo ALM** is a sophisticated, PyQt5-based algorithmic trading bot designed for MetaTrader 5. It utilizes an ensemble of machine learning models (Gradient Boosting, Random Forest, MLP) and custom logic engines ("Quantum", "Nexus", "GenZig") to execute high-probability scalping and swing trades.

![Status](https://img.shields.io/badge/Status-Beta-blue)
![Platform](https://img.shields.io/badge/Platform-MetaTrader5-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## ⚡ Key Features

* **Multi-Model AI Brain:** Uses an ensemble of `sklearn` classifiers to predict market direction.
* **Victory/Mistake Bank:** Reinforcement learning system that remembers past wins and losses to adapt logic.
* **Quantum Logic:** Analyzes Hurst Exponent and Entropy to detect market chaos vs. order.
* **Nexus Logic Engine:** Cross-validates AI signals with physics-based metrics before execution.
* **Aegis Risk Engine:** Dynamic position sizing and volatility-based risk management.
* **Real-time Visualization:** Built-in GUI using `PyQt5` and `pyqtgraph` for live chart and decision monitoring.

## 🛠️ Installation

### Prerequisites
1.  **MetaTrader 5 (MT5):** Installed and logged into a hedging account.
2.  **Python 3.8+**

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Paldo-ALM.git](https://github.com/YOUR_USERNAME/Paldo-ALM.git)
    cd Paldo-ALM
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  **MT5 Configuration:**
    * Ensure "Algo Trading" is enabled in MT5.
    * Add your script execution URL to "Allowed URLs" if using Telegram features.

## 🚀 Usage

1.  Open your MetaTrader 5 terminal.
2.  Run the bot:
    ```bash
    python main.py
    ```
3.  Login via the GUI popup using your Broker ID, Password, and Server.
4.  Select your execution mode (e.g., "Swing-Master", "Zenith Auto-Pilot").

## ⚙️ Architecture

* **ZenithBrain:** The core ML unit handling training and prediction.
* **EpochEngine:** Periodically optimizes AI weights based on recent performance.
* **GenZig Engine:** Backtests signals in real-time against recent history.
* **OmniMind:** The central strategy aggregator combining Technicals, AI, and Sentiment.

## ⚠️ Disclaimer

This software is for educational purposes only. Cryptocurrency and Forex trading involve significant risk. The authors are not responsible for any financial losses incurred while using this software.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
