"""Converts the Iris.csv file to other datatypes for testing"""
# Imports
import pandas as pd

# Read the df
df = pd.read_csv("test/dataloaders/data/Iris.csv")
# Json
df.to_json("test/dataloaders/data/Iris.json")
# Parquet
df.to_parquet("test/dataloaders/data/Iris.parquet")
# Excel
df.to_excel("test/dataloaders/data/Iris.xlsx", index=False)
