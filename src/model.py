import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

# adding dataset
data = pd.read_excel("./src/data/twiter_sentiment.xlsx")
df = pd.DataFrame(data)

# DATA PRE PROCESSING

# remove unused columns
df = df.drop(columns=["none", "none.1"])
# check null rows
print(df.isna().sum())
# remove rows with NaN values
df = df[df["Text"].notna()]

# data balancing chart
qtdSentiments = []
labels = df['Sentiment'].unique()
for i in labels:
    qtd = df['Sentiment'].value_counts()[i]
    qtd = int(qtd)
    qtdSentiments.append(qtd)
print(qtdSentiments)

# Remove - ''' - to show pie chart
'''fig, ax = plt.subplots()
ax.pie(qtdSentiments, labels=labels, autopct='%1.1f%%')
plt.show()''' 