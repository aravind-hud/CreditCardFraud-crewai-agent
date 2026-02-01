import os
from crewai.tools import BaseTool # Import from crewai.tools
from pydantic import Field
from typing import List
import pandas as pd



# --- STEP 1: Custom Tool for Windows/Python 3.12 ---
class FraudDataTool(BaseTool):
    name: str = "fraud_analysis_tool"
    description: str = "Use this to load the CSV, check for nulls, and calculate class imbalance."

    def _run(self, file_path: str) -> str:
        try:
            # Ensure path works on Windows
            df = pd.read_csv(file_path)
            stats = {
                "total_rows": len(df),
                "fraud_cases": int(df['Class'].sum()),
                "legit_cases": int(len(df) - df['Class'].sum()),
                "fraud_percentage": (df['Class'].sum() / len(df)) * 100,
                "columns": df.columns.tolist()
            }
            return f"Data Stats: {stats}"
        except Exception as e:
            return f"Error: {str(e)}"
