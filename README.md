# ML: Homework 2

## Team №
- Дрибноход Евгений Александрович, БАСБ251  
- Бакаленко Павел Игоревич, БАСБ251  
- Загреков Кирилл Андреевич, БАСБ251  
- Ворожцов Пётр Алексеевич, БАСБ252  

## About the Project
The goal of the project is to develop an interactive dashboard for analyzing the impact of changes in categorical and numerical parameters on the prediction results of ML models using marimo. The work is carried out as part of the course *Machine Learning for Business Analytics*.

## Project Demo
[**dribnokhod.com:8888**](http://dribnokhod.com:8888)

## About the Dataset
The project is based on **All Computer Prices**, which contains technical parameters and retail prices of computers. Errors related to incorrect operating system specifications for Apple devices (OS: Windows → macOS) have been corrected in the dataset.

Workflow:
1. Primary and exploratory data analysis (EDA) using Python.  
2. ML prediction of the target value (price).  
3. Development of an interactive dashboard for assessing the impact of parameter changes on prediction results using marimo.  

## Repository Structure
- `computer_prices_clean.csv` — final cleaned dataset.  
- `ML_HW_1.ipynb` — code for the first homework assignment with description.  
- `ML_HM_2.py` — code for the second homework assignment.  
- `` — presentation of the second project.  
- `README.md` — this file with a general project description.  

## Tools Used
- Python (pandas, numpy, sklearn, marimo) — data analysis, hypothesis testing.  

## Outcome
The project demonstrates the full cycle of analytical work: data processing, data analysis, prediction of target metrics, and visualization of results in a dashboard. The developed dashboard allows for interactive data exploration, revealing changes in predicted device prices depending on changes in technical parameters.
