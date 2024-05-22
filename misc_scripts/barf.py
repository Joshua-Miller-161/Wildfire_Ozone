import sys
sys.dont_write_bytecode = True
import os
os.environ['USE_PYGEOS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from misc.plotting_utils import ShowYearMonth

# Example data (replace with your actual data)
x = np.arange(1, 1001)  # Days from start_date
y_data = np.random.rand(len(x))  # Example y_data

# Assuming start_date is known (replace with your actual start date)
start_date = datetime(2024, 1, 1)

# Convert days to dates
dates = [start_date + timedelta(days=int(d)) for d in x]

# Extract years from dates
years = [date.year for date in dates]

# Create the plot
fig, ax = plt.subplots(1,1,figsize=(10, 6))


ax.scatter(dates, y_data, label='y_data', marker='o')
ax.set_xlabel('Year')
ax.set_ylabel('y_data')
ax.set_title('Plot of y_data vs. Year')
ax.grid(True)

ShowYearMonth(ax, dates, fontsize=12, method=0, rotation=0)

ax.legend()
plt.tight_layout()
plt.show()