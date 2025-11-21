import pandas as pd
import openpyxl

data = pd.read_excel("./src/data/twiter_sentiment.xlsx")
df = pd.DataFrame(data)
print(df)