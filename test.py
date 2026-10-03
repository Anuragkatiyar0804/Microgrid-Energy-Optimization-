import numpy
import pandas
import torch
import matplotlib
from data.loader import load_csv

print("Python environment working")


df = load_csv("data/raw/microgrid.csv")

print(df.head())
print(df.shape)