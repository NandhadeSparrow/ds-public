import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')
party_cnt = presidents_df['party'].value_counts()

plt.style.use('ggplot')
party_cnt.plot(kind ='bar')
plt.savefig("plot.png")
plt.show()



# bar plot
def addlabels(x, y, labels):
  for i in range(len(x)):
    plt.text(i, y[i], labels[i], ha = 'center', va = 'bottom')

plt.figure(figsize = (15, 5))
plt.bar(x = df_by_q['week'], height = df_by_q['total'])
addlabels(df_by_q['week'], df_by_q['total'], df_by_q['total'])
plt.plot()
plt.xlabel('Week')
plt.ylabel('Total Earn')
plt.title("Weekly Earn")
plt.xticks(rotation = 45, fontsize = 8)
plt.show()
plt.figure(figsize = (15, 5))
p = sns.barplot(
    data = df_by_w,
    x = 'week',
    y = 'total',
    hue = 'week',
    order = 'weekday_order',
    showfliers=False)
for b in p.patches:
  p.annotate(str(round(b.get_height()/1000, 1))+'k',
             (b.get_x()+b.get_width()/2., b.get_height()),
             ha = 'center', va = 'bottom', xytext = (0, -12),
             textcoords = 'offset points')
plt.xlabel('Week')
plt.ylabel('Total Earn')
plt.title("Weekly Earn")
plt.xticks(rotation = 45, fontsize = 8)
plt.show()