import numpy as np #importing relevant libraries
import pandas as pd
import csv

df = pd.read_csv('TitanicTrain.csv')
df.Age.fillna(value=df.Age.mean(), inplace=True) #filling NAN values with mean
df.Fare.fillna(value=df.Fare.mean(), inplace=True) # ^^
df.Embarked.fillna(value=(df.Embarked.value_counts().idxmax()), inplace=True) #filling NAN values with most popular category
df.Sex = df.Sex.apply(lambda x: 0 if x =="male" else 1) #changing sex to 0 and 1 for male and female respectively
df.Age = df.Age.apply(lambda x: x/df.Age.max())
df.Fare = df.Fare.apply(lambda x: x/df.Fare.max())
df.Embarked = df.Embarked.apply(lambda x: 1 if x == "C" else (2 if x == "Q"  else 3)) #changing embarked to numerical
family_size = pd.DataFrame(df.apply(lambda x: x.SibSp+x.Parch, axis=1), columns=["FamilySize"]) #creating family size
#creating title column by splitting the name
df = df.join(family_size) #adding family size column
isAlone = pd.DataFrame(df.apply(lambda x: 1 if x.FamilySize==0 else 0, axis=1), columns=["TravellingAlone"])
isAlone.TravellingAlone.value_counts()
df = df.join(isAlone)
titles = pd.DataFrame(df.apply(lambda x: x.Name.split(", ")[1].split(".")[0], axis=1), columns=["Title"]) 
df = df.join(titles)
df.Title = df.Title.replace(['Dr', 'Rev','Major','Col','Don','Lady','Sir','Capt','Jonkheer','Mme','the Countess'], 'Special')
df.Title = df.Title.replace(['Mlle','Ms'], 'Miss')
df = pd.concat([pd.get_dummies(df['Title'], prefix='Title'), df], axis=1)
df = pd.concat([pd.get_dummies(df['Embarked'], prefix='Embarked'), df], axis=1)
df = pd.concat([pd.get_dummies(df['Pclass'], prefix='Pclass'), df], axis=1)
df.drop('Name', axis=1, inplace=True) #dropping  irrelevant columns
df.drop('Cabin', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
#df.drop('Parch', axis=1, inplace=True)
#df.drop('SibSp', axis=1, inplace=True)
df.drop('Embarked', axis=1, inplace=True)
df.drop('TravellingAlone', axis=1, inplace=True)
df.drop('FamilySize', axis=1, inplace=True)
df.drop('Sex', axis=1, inplace=True)
df.drop('PassengerId', axis=1, inplace=True)
df.drop('Title', axis=1, inplace=True)
df.drop('Pclass', axis=1, inplace=True)
survivedCol = df.pop("Survived") #putting survived as the last column
df["Survived"] = survivedCol
df.to_csv('clean_titanic.csv', index=False)