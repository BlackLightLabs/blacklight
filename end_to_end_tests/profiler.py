from ydata_profiling import ProfileReport, model
import pandas as pd
data = pd.read_csv('data/Iris.csv', index_col=0)
profile = ProfileReport(data, title='Iris Dataset')
profile.to_file('output.json')