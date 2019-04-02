Titanic and Abalone Machine learning Project

This repository contains programs used to clean, visualise and 
fit machine learning models to the Titanic Data set from Kaggle:
https://www.kaggle.com/c/titanic
and the UCI Abalone dataset:
https://archive.ics.uci.edu/ml/datasets/abalone
The programs are coded in python. The directory "FOMLADS" was 
provided in a course taken at UCL and is needed for the fisher 
discriminant and logistic regression programs. sklearn, numpy, 
pandas, and seaborn are needed.

*******************************************************************
Make sure that you have python 3.7 and the following imports installed to your system:
matplotlib, numpy, pandas, seaborn and sklearn
*******************************************************************

There are 2 datasets in the folder, clean_abalone.csv and clean_titanic.csv.

Open terminal
Using terminal command line, navigate to the folder that is unzipped from the zip file.
This ensures the correct working directory.

*******************************************************************
Fisher’s Linear Discriminant

To run Fisher’s Linear Discriminant experiment on Titanic dataset, enter this command line in terminal: 

python3 Fishers.py clean_titanic.csv                                                

or, for abalone dataset

python3 Fishers.py clean_abalone.csv

You should see 6 figures for each run: 

2 figures (normal dataset and transformed dataset) on how changes in decision boundary changes the training/ testing prediction accuracy; 

2 figures (normal dataset and transformed dataset) on confusion matrices;

1 figure shows the receiver operating characteristic curve (ROC) plot with the values of area under curve (AUC) and the corresponding dataset (normal and transformed dataset) at the bottom right of the graph and

1 figure on the results of the models (such as precision, sensitivity and specificity etc.).

*******************************************************************
Logistic Regression

To run Logistic Regression experiment on Titanic dataset, enter this command line in terminal: 

python3 Logistic_Regression.py clean_titanic.csv                                                

or, for abalone dataset

python3 Logistic_Regression.py clean_abalone.csv

You should see 4 figures for each run:

2 figures (normal dataset and transformed dataset) on confusion matrices;

1 figure on ROC curve plot and

1 figure on the results of the models.
*******************************************************************
K Nearest Neighbours

To run K Nearest Neighbours experiment on Titanic dataset, enter this command line in terminal: 

python3 KNN.py clean_titanic.csv                                                

or, for abalone dataset

python3 KNN.py clean_abalone.csv

You should see 5 figures for each run:

2 figures (normal dataset and transformed dataset) on confusion matrices;

1 figure on cross validation;

1 figure on ROC curve plot and

1 figure on the results of the models.
