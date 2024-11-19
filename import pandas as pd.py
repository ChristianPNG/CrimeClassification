import matplotlib.pyplot as plt
import pandas as pd

# Given attributes
attributes = ['Occ_Year', 'Occ_Month', 'Occ_Day', 'Rptd_Year', 'Rptd_Occ_Month', 'Rptd_Occ_Day',
              'TIME OCC', 'AREA', 'Rpt Dist No', 'Part 1-2', 'Mocodes_First4', 'Premis Cd', 'LOCATION',
              'Vict Age', 'Vict Sex', 'Vict Descent', 'Weapon Used Cd', 'Status', 'LAT', 'LON']

# Create a DataFrame with a single column
df = pd.DataFrame(attributes, columns=['Attributes'])

# Add a column for numbering
df['Number'] = range(1, len(attributes) + 1)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(2, 8))
ax.axis('off')  # Turn off the axis

# Create the table
table = ax.table(cellText=df[['Number', 'Attributes']].values, cellLoc='center', loc='center', colLabels=['', 'Attributes'])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(.4, 1.2)  # Adjust the table size

# Show the plot
plt.show()
