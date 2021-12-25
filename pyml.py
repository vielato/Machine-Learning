# Load CSV using Pandas from URL
import os
import pandas
import matplotlib
import pandas as pd
filename = 'dataset/BTC-USD.csv'
data = pd.read_csv(filename)
data.plot.line(y="Close", x="Date")
matplotlib.pyplot.show()


