from dotenv import load_dotenv
from crewai import Crew
from tasks import FraudDataAnalystTasks
from agents import FraudDataAnalyst
import os


def main():
    load_dotenv()
    
    print("## Welcome to the Data Analysis project")
    print('-------------------------------')
   

    tasks = FraudDataAnalystTasks()
    agents = FraudDataAnalyst()
    
    # create agents
    data_loader_agent = agents.data_loader_agent()
    data_analyst_agent = agents.data_analyst_agent()
    ml_engineer_agent = agents.ml_engineer_agent()
    prediction_agent = agents.prediction_agent()
    
    # create tasks
    loading_task = tasks.loading_task(data_loader_agent)
    analysis_task = tasks.analysis_task(data_analyst_agent)
    modeling_task = tasks.modeling_task(ml_engineer_agent)
    prediction_task = tasks.prediction_task(prediction_agent)
    
    modeling_task.context = [loading_task, analysis_task]
    prediction_task.context = [loading_task,analysis_task, modeling_task]
    
    crew = Crew(
      agents=[
        data_loader_agent,
        data_analyst_agent,
        ml_engineer_agent,
        prediction_agent
      ],
      tasks=[
        loading_task,
        analysis_task,
        modeling_task,
        prediction_task
      ]
    )
    
    result = crew.kickoff()
    
    print(result)
    
if __name__ == "__main__":
    main()
