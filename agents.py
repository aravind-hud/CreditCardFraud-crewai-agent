from textwrap import dedent
from crewai import Agent, LLM
from tools import FraudDataTool
import os

class FraudDataAnalyst():
    # No __init__ with ChatOpenAI needed if using string identifiers
    def __init__(self):
        # Initialize Gemini LLM
        self.gemini_llm = LLM(
        model=os.getenv("MODEL"), # Note the 'gemini/' prefix
        api_key=os.getenv("GEMINI_API_KEY")
        )

    def data_loader_agent(self):
        return Agent(
        role='Data Engineer',
        goal='Load and validate the integrity of the credit card transaction data.',
        backstory='Expert in ETL and data cleaning, ensuring no missing values disrupt the ML pipeline.',
        tools=[FraudDataTool()],
        llm=self.gemini_llm,
        verbose=True
        )

    def data_analyst_agent(self):
        return Agent(
        role='Fraud Data Analyst',
        goal='Identify patterns in fraudulent transactions and compare XGBoost vs Other models.',
        backstory='Specialist in high-imbalance datasets. You understand why SMOTE or Class Weights are needed.',
        llm=self.gemini_llm,
        verbose=True
        )
    
    # ... apply the same llm="gpt-3.5-turbo" to other agents
      
    def ml_engineer_agent(self):
        return Agent(
        role='ML Implementation Specialist',
        goal='Design the final XGBoost architecture and prediction strategy.',
        backstory='You turn theories into Python code, focusing on precision-recall curves over simple accuracy.',
        llm=self.gemini_llm,
        verbose=True
        )
      
    def prediction_agent(self):
        return Agent(
        role='Model building for prediction',
        goal='Your goal is to build a prediction model specifically using the data loaded via the FraudDataTool from the local creditcard.csv. You must never use simulated or synthetic data in your code output. Every code snippet must begin with loading the actual dataset provided',
        backstory=dedent("""\
            You are an expert data scientist and ML engineer and good understanding of Credit Card Fraud. As the prediction analyst, your role is to consolidate the research,
            analysis, and modeling insights and build a robust prediction model that works well with this data."""),
        llm=self.gemini_llm,
        verbose=True,
        ###allow_code_execution=false,
        ###code_execution_mode="local"
        )
