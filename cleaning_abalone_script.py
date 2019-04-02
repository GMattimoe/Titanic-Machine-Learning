import numpy as np #importing relevant libraries
import pandas as pd
import csv

df = pd.read_csv('abalone.txt', header=None, names=["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"])
df.Sex = df.Sex.apply(lambda x: 0 if x =="M" else 1) #converting sex to 0 and 1 for male and female respectively
df.Rings = df.Rings.apply(lambda x: x+1.5) #adding 1.5 to ring for age
df = df.rename(columns={'Rings': 'Old'})
df.Old = df.Old.apply(lambda x: '1' if x>10 else '0') #changing age to 0 or 1 for younger or older than 10 years

df.to_csv('clean_abalone.csv', index=False)