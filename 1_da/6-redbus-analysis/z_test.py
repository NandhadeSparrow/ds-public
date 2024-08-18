import numpy as np 
import matplotlib.pyplot as plt 
import streamlit as st

N = 3
ind = np.arange(N) 
width = 0.25

xvals = [8, 9, 2] 
bar1 = plt.bar(ind, xvals, width, color='r') 

yvals = [10, 20, 30] 
bar2 = plt.bar(ind+width, yvals, width, color='g') 

zvals = [11, 12, 13] 
bar3 = plt.bar(ind+width*2, zvals, width, color='b') 

plt.xlabel("Dates") 
plt.ylabel('Scores') 
plt.title("Players Score") 

plt.xticks(ind+width, ['2021Feb01', '2021Feb02', '2021Feb03']) 
plt.legend((bar1, bar2, bar3), ('Player1', 'Player2', 'Player3'))

# Display the plot in Streamlit
st.pyplot(plt)


'''

Uttar Pradesh,Aligarh (uttar pradesh) to Delhi,https://www.redbus.in/bus-tickets/aligarh-uttar-pradesh-to-delhi
South Bengal,Barasat (West Bengal) to Digha,https://www.redbus.in/bus-tickets/barasat-west-bengal-to-digha
Telangana,Karimnagar to Hyderabad,https://www.redbus.in/bus-tickets/karimnagar-to-hyderabad
Telangana,Hyderabad to Nirmal,https://www.redbus.in/bus-tickets/hyderabad-to-nirmal
Bihar,Ranchi to Muzaffarpur (Bihar),https://www.redbus.in/bus-tickets/ranchi-to-muzaffarpur
Bihar,Motihari to Lucknow,https://www.redbus.in/bus-tickets/motihari-to-lucknow
Himachal,Manali to Delhi,https://www.redbus.in/bus-tickets/manali-to-delhi
Himachal,Hamirpur (Himachal Pradesh) to Delhi,https://www.redbus.in/bus-tickets/hamirpur-himachal-pradesh-to-delhi


'''