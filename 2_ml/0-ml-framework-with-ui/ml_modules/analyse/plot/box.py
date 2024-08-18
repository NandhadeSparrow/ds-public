import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')
plt.style.use('classic')
presidents_df.boxplot(column='height');
plt.savefig("plot.png")
plt.show()


# boxplot
box = sns.boxplot(x=df['number_of_strikes'])
g = plt.gca()
box.set_xticklabels(np.array([readable_numbers(x) for x in g.get_xticks ()]))
plt.xlabel('Number of strikes')
plt.title('Yearly number of lightning strikes')