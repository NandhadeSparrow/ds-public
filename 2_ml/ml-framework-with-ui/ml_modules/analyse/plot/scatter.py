import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')


plt.scatter(presidents_df['height'], presidents_df['age'],
   marker='<',
   color='b')
plt.xlabel('height'); 
plt.ylabel('age')
plt.title('U.S. presidents')


presidents_df.plot(kind = 'scatter', 
  x = 'height', 
  y = 'age',
  title = 'U.S. presidents')


plt.savefig("plot.png")
plt.show()

import matplotlib.pyplot as plt
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

plt.scatter(df['Age'], df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])

plt.plot([0, 80], [85, 5])



# scatter plot
import plotly.express as px

fig = px.scatter_geo = (top_missing[number_of_strikes_x >= 300],
                        lat='lattitude',
                        lon='longitude',
                        size='number_of_strikes_x')
fig.update_layout(title_text = 'title', geo_scope='usa')
fig.show()