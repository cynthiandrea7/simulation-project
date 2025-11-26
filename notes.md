reducing the number of companies, find a subset of each sector to simpify the problem
- stratified by sector

look at the weights how the code pulling it was constructed
- set different thredholds
- set X weights and drop the rest deending on how diversified the project (main focus) is

objectives:
- differnet strategies of constructing a portfolio
    - minimizing risk (SD over time)
        - if going after diversified -- we need coorletation 
        - max % of risk
    - maximizing profit
    - a balanced strategy? 
same budget 
finding a measure of the risk for each profile

can frame the optimization (max expected return) following the 3 different stragerties 

--- 


optimization problem
- risk analysis
- sensitivity analysis


## Project Outline

to-do item:
- Mint to add the weight column to the sp500_sectors.csv

# Introduction

# RQ and Objectives

given the same budget, how would you construct an investment portfolio / allocate funds to different stocks tailored to different investor profiles that maximize profits and satisfies their risk tolerance %, based on the last 5 years of historical data.


objectives:
- differnet strategies of constructing a portfolio
    - minimizing risk (SD over time)
        - if going after diversified -- we need coorletation 
        - max % of risk
    - maximizing profit
    - a balanced strategy? 
same budget 
finding a measure of the risk for each profile

profile:
1. aggressive to maximize returns, high risk tolerence
2. balanced to maximize returns and balance between risk
3. conservative to maximize returns and minimize risk

all profiles will have the same budget

each of the profile will have a different risk contraints (%)

# Model Construction

## EDA

- reducing the number of companies, find a subset of each sector to simpify the problem
- stratified by sector
- dataset: more representative, we want to have all industries
-- within each industry, we want to keep 3 companies (33 variables in total)
-- option 1: we can go by the weights, select the companies with the higest weights in each sector


-- do a EDA on the subset

## 

- analyze the returns of each company 
- analyze the risk of each stock (SD)
- optimization based on the returns and risks for each profile
- run some sensitivity analysis to each of the portfolio, what happens if we change how much risk i'm willing to take on

--- 

use the historical variance / sd to measure the risk

(to take this to the next level: use time-series and simulate)
# right now we just average them out over 5 years
# alternatively we can assign more weights to recent years
