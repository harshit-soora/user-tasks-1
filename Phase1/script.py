import pandas as pd

# Load the dataset
data = pd.read_csv('dataset.csv')

# Sort by sale price in descending order and get the top 2 properties
top_properties = data.sort_values(by='Sale Price', ascending=False).head(2)
# Parse "Sale Date" to extract the year and filter for year 2022
data['Year'] = pd.to_datetime(data['Sale Date']).dt.year
data_2022 = data[data['Year'] == 2022]

# Sort by sale price in descending order and get the top 2 properties
top_properties = data_2022.sort_values(by='Sale Price', ascending=False).head(2)

# Display the result
print(top_properties)
top_properties.to_csv('result.txt', index=False)