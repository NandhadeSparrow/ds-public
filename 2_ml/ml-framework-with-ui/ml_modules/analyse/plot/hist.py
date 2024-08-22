import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')
presidents_df['height'].plot(kind='hist',
  title = 'height',
  bins=5)
plt.savefig("plot.png")
plt.show()

