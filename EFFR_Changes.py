#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: shawnbrar
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

#Read 5 csv files

XLK_Tech_Data = pd.read_csv('/Users/shawnbrar/Desktop/Hofstra_Classes/DS_221/XLK_Tech_Data.csv', skiprows = [1,2])
#Have to skip importing rows 1 and 2 b/c they contain unnecessary data that arent part of the actual data
#print(XLK_Tech_Data.head())

XLV_Healthcare_Data = pd.read_csv('/Users/shawnbrar/Desktop/Hofstra_Classes/DS_221/XLV_Healthcare_Data.csv', skiprows = [1,2])
#print(XLV_Healthcare_Data.head())

XLF_Finance_Data = pd.read_csv('/Users/shawnbrar/Desktop/Hofstra_Classes/DS_221/XLF_Finance_Data.csv', skiprows = [1,2])
#print(XLF_Finance_Data.head())

XLY_Consumer_Data = pd.read_csv('/Users/shawnbrar/Desktop/Hofstra_Classes/DS_221/XLY_Consumer_Data.csv', skiprows = [1,2])
#print(XLY_Consumer_Data.head())

#FFER_Data = pd.read_csv('/Users/shawnbrar/Desktop/DS_221/FFER_Dataset.csv')
#print(FFER_Data.head())
#FFER_Data.rename(columns = {'FEDFUNDS':'EFFR','DATE':'Date'}, inplace = True) #Rename column Date so all datasets can be merged on one column with the same name
#FFER_Data['Date'] = pd.to_datetime(FFER_Data['Date']).dt.date #make sure all datasets have same date format

###ORGANIZING/CLEANING STOCK DATA

##Organize the XLK Data

XLK_Tech_Data.rename(columns = {'Price':'Date'}, inplace = True)
#rename Price column to Date b/c data was imported incorrectly
#use inplace = True so I dont have to make another variable of the data


#Fix the date so it alligns with the FFER_Data and standardized as a simple data format YY-MM-DD
XLK_Tech_Data['Date'] = pd.to_datetime(XLK_Tech_Data['Date']).dt.date


#Drop all unnecesary columns, we only want to look at the stock Close column to analyze stock performance over time
drop_unnecesary_columns = ['Adj Close', 'High', 'Low', 'Open', 'Volume']
XLK_Tech_Data = XLK_Tech_Data.drop(columns = drop_unnecesary_columns)

#Rename Close column to Close(Tech) to make it easier to identify which industry 
XLK_Tech_Data.rename(columns = {'Close':'Close(Tech)'}, inplace = True) 
#print(XLK_Tech_Data.head())


#Now I repeat the same process for the XLY, XLF, and XLV stock data

XLV_Healthcare_Data.rename(columns = {'Price':'Date'}, inplace = True)
XLV_Healthcare_Data['Date'] = pd.to_datetime(XLV_Healthcare_Data['Date']).dt.date
XLV_Healthcare_Data = XLV_Healthcare_Data.drop(columns = drop_unnecesary_columns)
XLV_Healthcare_Data.rename(columns = {'Close':'Close(Healthcare)'}, inplace = True) 
#print(XLV_Healthcare_Data.head())

XLF_Finance_Data.rename(columns = {'Price':'Date'}, inplace = True)
XLF_Finance_Data['Date'] = pd.to_datetime(XLF_Finance_Data['Date']).dt.date
XLF_Finance_Data = XLF_Finance_Data.drop(columns = drop_unnecesary_columns)
XLF_Finance_Data.rename(columns = {'Close':'Close(Finance)'}, inplace = True) 
#print(XLF_Finance_Data.head())

XLY_Consumer_Data.rename(columns = {'Price':'Date'}, inplace = True)
XLY_Consumer_Data['Date'] = pd.to_datetime(XLY_Consumer_Data['Date']).dt.date
XLY_Consumer_Data = XLY_Consumer_Data.drop(columns = drop_unnecesary_columns)
XLY_Consumer_Data.rename(columns = {'Close':'Close(Consumer)'}, inplace = True) 
#print(XLY_Consumer_Data.head())

###MERGE all the datasets into one dataset (merge on 'Date')

merged_datasets = pd.merge(FFER_Data, XLK_Tech_Data, on='Date')
merged_datasets = pd.merge(merged_datasets, XLV_Healthcare_Data, on = 'Date')
merged_datasets = pd.merge(merged_datasets, XLF_Finance_Data, on = 'Date')
merged_datasets = pd.merge(merged_datasets, XLY_Consumer_Data, on ='Date')

print(merged_datasets.head())

#saving my dataset as a file
#Merged_Stock_EFFR_Data = merged_datasets
#Merged_Stock_EFFR_Data.to_csv('/Users/shawnbrar/Desktop/DS_221/Merged_Stock_EFFR_Data.csv')

#Check to see if data is clean 


print("\n")

print("Cheking to see if the data contains any null values: ")
print(merged_datasets.isnull().any())

print("\n")
print('Shape of data: ')
print(merged_datasets.shape)

print("\n")
print('Data Types: ')
merged_datasets['Date'] = pd.to_datetime(merged_datasets['Date'], errors='coerce') 
#python was having trouble with the '-' in between the dates so by using coerce pandas will ignore them and convert the dtype to datetime instead of object
print(merged_datasets.dtypes)

###Part 3 (EDA)
## 1. Summary Statisitcs 


summary_statistics = merged_datasets.describe() 
#gets mean, median, sd, min, max, quartiles 
numeric_columns = merged_datasets.select_dtypes(include = 'float64') #selecting all columns except 'Date'


var = numeric_columns.var()  #getting variance of columns
summary_statistics.loc['var'] = var
summary_statistics = summary_statistics.drop(columns = 'Date') #drop 'Date' as it is unnecessary 

print("\n")
print("Summmary Statistics: ")
print(summary_statistics) 

## 2. Data Visualization/Correlation Analysis

#Line graph of EFFR over time (see how the interst rate changed over 56 months)

x = merged_datasets.Date 
y = merged_datasets.EFFR
plt.figure(figsize = (13,10))
plt.title('Effective Federal Funds Rate Over Time (Jan 2020 - August 2024)')
plt.xlabel('Date')
plt.ylabel('Effective Federal Funds Rate')
plt.plot(x,y)
plt.show()

#Scatterplot relationship between EFFR & Tech Sector

effr_x = merged_datasets['EFFR']
y_tech = merged_datasets['Close(Tech)']
plt.figure(figsize = (13,10))
plt.xlabel("Effective Federal Funds Rate", fontsize = 17)
plt.ylabel("Closing Price (Technology Sector)", fontsize = 17)
plt.title('Relationship Between EFFR and Technology Sector Closing Prices', fontsize = 20, pad = 5, )
plt.scatter(effr_x,y_tech, color = 'royalblue', s = 120)
plt.grid(linewidth = 0.5)
plt.show()

effr_tech_correlation = merged_datasets[['EFFR','Close(Tech)']].corr()
print("\n")
print('Correlation beween EFFR & Close(Tech): ')
print(effr_tech_correlation) 

#Scatterplot relationship between EFFR & Health sector


y_healthcare = merged_datasets['Close(Healthcare)']
plt.figure(figsize = (13,10))
plt.xlabel("Effective Federal Funds Rate", fontsize = 17)
plt.ylabel("Closing Price (Healthcare Sector)", fontsize = 17)
plt.title('Relationship Between EFFR and Healthcare Sector Closing Prices', fontsize = 20, pad = 5, )
plt.scatter(effr_x,y_healthcare, color = 'royalblue', s = 120)
plt.grid(linewidth = 0.5)
plt.show()

effr_health_correlation = merged_datasets[['EFFR','Close(Healthcare)']].corr()
print("\n")
print('Correlation beween EFFR & Close(Healthcare): ')
print(effr_health_correlation) 


#Scatterplot relationship between EFFR & Finance sector 
y_finance = merged_datasets['Close(Finance)']
plt.figure(figsize = (13,10))
plt.xlabel("Effective Federal Funds Rate", fontsize = 17)  
plt.ylabel("Closing Price (Finance Sector)", fontsize = 17)
plt.title('Relationship Between EFFR and Finance Sector Closing Prices', fontsize = 20, pad = 5, )
plt.scatter(effr_x,y_finance, color = 'royalblue', s = 120) 
plt.grid(linewidth = 0.5)
plt.show()

effr_finance_correlation = merged_datasets[['EFFR','Close(Finance)']].corr()
print("\n")
print('Correlation beween EFFR & Close(Finance): ')
print(effr_finance_correlation) 


#Scatterplot relationship between EFFR & consumer discretionary 

y_consumer = merged_datasets['Close(Consumer)']
plt.figure(figsize = (13,10))
plt.xlabel("Effective Federal Funds Rate", fontsize = 17)
plt.ylabel("Closing Price (Consumer Discretionary Sector)", fontsize = 17) 
plt.title('Relationship Between EFFR and Consumer Sector Closing Prices', fontsize = 20, pad = 5, )
plt.scatter(effr_x,y_consumer, color = 'royalblue', s = 120) 
plt.grid(linewidth = 0.5)
plt.show()

effr_consumer_correlation = merged_datasets[['EFFR','Close(Consumer)']].corr()
print("\n")
print('Correlation beween EFFR & Close(Consumer): ')
print(effr_consumer_correlation) 

##Box plot distribution over sectors
plt.boxplot([merged_datasets['Close(Tech)'],merged_datasets['Close(Healthcare)'],merged_datasets['Close(Finance)'],merged_datasets['Close(Consumer)']],
            labels = ['Tech','Healthcare', 'Finance', 'Consumer'])
plt.title('Distributions Of Closing Prices By Sector')
plt.ylabel('Closing Price ($ USD)')
plt.show()

# 3. Data Distribution:
    
#Histograms that describe the distribution of the 5 variables

#Histogram for EFFR

plt.hist(merged_datasets['EFFR'], edgecolor = 'black', color = 'royalblue')
plt.xlabel('EFFR')
plt.ylabel('Number Of Months')
plt.title('Distribution Of EFFR')
plt.show()

#Histogram for Close(Tech)

plt.hist(merged_datasets['Close(Tech)'], edgecolor = 'black', color = 'royalblue')
plt.xlabel('Close(Tech)')
plt.ylabel('Number Of Months')
plt.title('Distribution Of Closing Prices In Tech Sector')
plt.show()


#Histogram for Close(Healthcare)

plt.hist(merged_datasets['Close(Healthcare)'], edgecolor = 'black', color = 'royalblue')
plt.xlabel('Close(Healthcare)')
plt.ylabel('Number Of Months')
plt.title('Distribution Of Closing Prices In Healthcare Sector')
plt.show()

#Histogram for Close(Finance)

plt.hist(merged_datasets['Close(Finance)'], edgecolor = 'black', color = 'royalblue')
plt.xlabel('Close(Finance)')
plt.ylabel('Number Of Months')
plt.title('Distribution Of Closing Prices In Financial Sector')
plt.show()

#Histogram for Close(Consumer)

plt.hist(merged_datasets['Close(Consumer)'], edgecolor = 'black', color = 'royalblue')
plt.xlabel('Close(Consumer)')
plt.ylabel('Number Of Months')
plt.title('Distribution Of Closing Prices In Consumer Discretionary Sector')
plt.show()

#Hypothesis Testing

#Take mean of 100 random samples from EFFR 10000 times and then append to EFFR_mean_size_100
#This allows for an approximation of a normal dist as the sample size gets larger 

EFFR_mean_size_100 = []
for i in range (10000):
    sample_effr = np.random.choice(merged_datasets['EFFR'],100).mean()
    EFFR_mean_size_100.append(sample_effr)
    
#Now doing the same thing for Close(Consumer) to take advantage of the Central Limit Theorem

Consumer_mean_size_100 = []
for i in range(10000):
    sample_consumer = np.random.choice(merged_datasets['Close(Consumer)'],100).mean()
    Consumer_mean_size_100.append(sample_consumer)
    
    
#Histogram for EFFR after taking multiple random samples and averaging them to use the CLT    
plt.hist(EFFR_mean_size_100, edgecolor = 'black', color = 'royalblue') 
plt.xlabel('EFFR')
plt.ylabel('Number Of Months')
plt.title('Distribution Of EFFR (Normalized)')
plt.show()

#Histogram for Close(Consumer) after taking multiple random samples and averaging them to use the CLT

plt.hist(Consumer_mean_size_100, edgecolor = 'black', color = 'royalblue') 
plt.xlabel('Close(Consumer)')
plt.ylabel('Number Of Months')
plt.title('Distribution Of Closing Prices In Financial Sector (Normalized)')
plt.show()

#Now performing independent ttest

#Have to split the EFFR into two groups: High Effr group & Low EFFR group
#spltiing at the mean

average_EFFR = merged_datasets['EFFR'].mean()
above_average_EFFR = merged_datasets[merged_datasets['EFFR'] > average_EFFR]['Close(Consumer)'] 
below_average_EFFR = merged_datasets[merged_datasets['EFFR'] <= average_EFFR]['Close(Consumer)'] 
#splitting dataset into two different groups
#gives us the closing price of the consumer discretionary stock when Effer is above/below average

#ttest

statistic, p_value = stats.ttest_ind(a = above_average_EFFR, b = below_average_EFFR, equal_var = False)
print("\n")
print('HYPOTHESIS TESTING')
print("t_stat: ", statistic, "p_value: ", p_value) 

alpha = .05
if pvalue < .05:
    print("Failed to reject the null hypothesis")
else:
    print("Reject the null hypothesis, the result is statistically significant at the 5% significance level")
    



### Part 4 Machine Learning 

#Decision Tree Regressor Model
#Predicting performance of the tech sector by predicting the closing stock price

#features and target variable
x = merged_datasets.iloc[:,[1]].values
y = merged_datasets.iloc[:,[2]].values

#split data so 20% is for testing and 80% is for training

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

decision_tree_squared_error_model = DecisionTreeRegressor(criterion = 'squared_error', random_state = 42) #regression dt model uses squared_error as the criterion, not gini or entropy
decision_tree_squared_error_model.fit(x_train,y_train) 
y_pred_decision_tree_squared_error_model = decision_tree_squared_error_model.predict(x_test) 

decision_tree_squared_error_mse = mean_squared_error(y_test,y_pred_decision_tree_squared_error_model) #get mse


decision_tree_squared_error_R2_score = r2_score(y_test, y_pred_decision_tree_squared_error_model) #get R^2 of model


print('\n')  
print("Predicted Array Of Tech Sector Performance: ", y_pred_decision_tree_squared_error_model)
print('\n')  
print("Mean Squared Error Of Decision Tree Regression Model: ",decision_tree_squared_error_mse )
print("R2 Score of Decision Tree Model Using Squared Error: ", decision_tree_squared_error_R2_score)

#plotting decision tree
plt.figure(figsize = (25,20), dpi = 100)
plot_tree(decision_tree_squared_error_model, feature_names=['EFFR'], filled = True)
plt.title("Decision Tree for Tech Sector Closing Price Prediction")
plt.savefig('/Users/shawnbrar/Desktop/DS_221/tech_sector_decision_tree.png', dpi = 100)
plt.show()
plt.close



#Predicting performance of the healthcare sector by predicting the closing stock price 

x = merged_datasets.iloc[:,[1]].values
y_health = merged_datasets.iloc[:,[3]].values


#split data so 20% is for testing and 80% is for training

x_train, x_test, y_health_train, y_health_test = train_test_split(x,y_health,test_size = 0.2, random_state = 42)

decision_tree_squared_error_model = DecisionTreeRegressor(criterion = 'squared_error', random_state = 42) #regression dt model uses squared_error as the criterion, not gini or entropy
decision_tree_squared_error_model.fit(x_train,y_health_train) 
y_pred_decision_tree_squared_error_model = decision_tree_squared_error_model.predict(x_test) 

decision_tree_squared_error_mse = mean_squared_error(y_health_test,y_pred_decision_tree_squared_error_model) #get mse


decision_tree_squared_error_R2_score = r2_score(y_health_test, y_pred_decision_tree_squared_error_model) #get R^2 of model


print('\n')  
print("Predicted Array Of Healthcare Sector Performance: ", y_pred_decision_tree_squared_error_model)
print('\n')  
print("Mean Squared Error Of Decision Tree Regression Model: ",decision_tree_squared_error_mse )
print("R2 Score of Decision Tree Model Using Squared Error: ", decision_tree_squared_error_R2_score)


#plotting decision tree
plt.figure(figsize = (25,20), dpi = 100)
plot_tree(decision_tree_squared_error_model, feature_names=['EFFR'], filled = True)
plt.title("Decision Tree for Healthcare Sector Closing Price Prediction")
plt.savefig('/Users/shawnbrar/Desktop/DS_221/healthcare_sector_decision_tree.png', dpi = 100)
plt.show()
plt.close()



















































