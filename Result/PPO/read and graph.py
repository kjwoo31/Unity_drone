import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('PPO_drone_log_0.csv', sep = ",", engine='python', encoding = "utf-8")

df = df[df['episode']%100==0]

x = df['episode']
y = df['reward']

print(df[df['reward']>300])

# plot
plt.plot(x,y)
# beautify the x-labels
plt.gcf().autofmt_xdate()

plt.show()