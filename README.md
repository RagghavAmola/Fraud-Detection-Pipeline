# Financial Transaction Fraud Detection and Credit Risk Pipeline

The Financial Transaction Fraud Detection and Credit Risk Pipeline is an end-to-end data processing and analytics system designed to identify fraudulent financial transactions and evaluate credit risk in real-time and batch modes. Built with Python, SQLite, SQLAlchemy, Pandas, Streamlit, and Plotly, the pipeline ingests raw transaction feeds, applies domain-specific feature engineering, evaluates modular rule-based fraud detection heuristics and credit risk scoring mechanisms, persists structured records into a SQLite relational store, and surfaces actionable intelligence through an interactive monitoring dashboard.

## Project Structure

```text
fraud_pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingestion/
│   │   └── data_loader.py
│   ├── transformation/
│   │   └── feature_engineering.py
│   ├── rules_engine/
│   │   ├── rules_config.yaml
│   │   ├── rule_definitions.py
│   │   └── engine.py
│   ├── database/
│   │   ├── models.py
│   │   └── db_manager.py
│   └── utils/
├── dashboard/
│   └── app.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Prerequisites
- Python 3.10+ installed on your system.
- Git (optional, for version control).

### 2. Create and Activate Virtual Environment

**Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required pinned packages:
```bash
pip install -r requirements.txt
```

### 4. Running the Dashboard
Launch the interactive Streamlit monitoring dashboard:
```bash
streamlit run dashboard/app.py
```

### 5. Running Tests
Run the test suite:
```bash
pytest
```