import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_excel('/home/fan/Immunotherapy/MyData/AllData.xlsx')
print(df.head())
df['Gender'] = df['Gender'].map({'F': 1, 'M': 0})
df['Distance to Hospital'] = df['Distance to Hospital'].map({'>10km': 1, '<10km': 0})
df['ration of Treatment cost to Family income (Year)'] = df['ration of Treatment cost to Family income (Year)'].map(
    {'<30%': 0, '50%-70%': 1, '>70%': 2})
df.to_excel('/home/fan/Immunotherapy/MyData/AllData.xlsx')
