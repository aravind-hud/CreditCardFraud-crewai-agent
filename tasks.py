from textwrap import dedent
from crewai import Task

class FraudDataAnalystTasks():
  
  def loading_task(self, data_loader_agent):
    return Task(
    description="Load the dataset from 'creditcard.csv' and report on the fraud-to-legit ratio.",
    expected_output="A summary of the dataset shape and the severity of the class imbalance.",
    agent=data_loader_agent
    )
    
  def analysis_task(self, data_analyst_agent):
    return Task(
    description=(
        "Analyze why XGBoost might outperform Random Forest for this specific dataset. "
        "Suggest techniques to handle the 0.17% fraud rate."
    ),
    expected_output="A comparison report recommending XGBoost or an alternative, with specific reasoning.",
    agent=data_analyst_agent
    )
    
  def modeling_task(self, ml_engineer_agent):
    return Task(
    description="Provide a Python code blueprint for the recommended model using XGBoost.",
    expected_output="A full Python code snippet including data splitting and the XGBoost 'scale_pos_weight' parameter.",
    agent=ml_engineer_agent
    )
    
  def prediction_task(self, prediction_agent):
    return Task(
			description=dedent(f"""\
			(1.Using the context from the loading_task, generate a complete Python script. 
            The script must use pandas.read_csv('creditcard.csv') to load the data. Do not include blocks for data simulation. Ensure the code is ready to run against the local file directly..
            2. Recommend and build a specific machine learning model (e.g., XGBoost, LSTM) to predict fraud.
            3. Provide a step-by-step guide on how you build this model in python.
            4. provide clear information on how you evaluated the model - which metric and why you think this is good outcome"
            ),"""),
			expected_output=dedent("""\
				A well-structured code and document that includes sections for
				overview, analysis, insights, model strategy, implementation guidde, code and future recommendation."""),
			agent=prediction_agent
    )
