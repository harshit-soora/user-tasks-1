import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('dataset.csv')

# Current estimated value of a typical home in the area
average_home_value = data['Sale Price'].mean()
print(f"Current estimated value of a typical home: ${average_home_value:,.2f}")
print("\n")

# Best time to sell based on seasonal trends
data['Month'] = pd.to_datetime(data['Sale Date']).dt.month
monthly_avg = data.groupby('Month')['Sale Price'].mean()

plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values)
plt.title('Average Home Value by Month')
plt.xlabel('Month')
plt.ylabel('Average Home Value')
plt.xticks(range(1, 13))
plt.grid()
plt.show()

best_month = monthly_avg.idxmax()
print(f"The best time to sell based on seasonal trends is month: {best_month}")
print("\n")

# Home improvements that might yield the best return on investment
# Exclude non-numeric columns like 'Address' from correlation computation
numeric_data = data.select_dtypes(include=['number'])
improvement_correlation = numeric_data.corr()['Sale Price'].sort_values(ascending=False)
print("Correlation of home improvements with home value:")
print(improvement_correlation)

# Assuming columns like 'KitchenUpgrade', 'BathroomUpgrade', etc., exist
top_improvements = improvement_correlation[1:4]  # Top 3 improvements
print("Top home improvements for best ROI:")
print(top_improvements)

###
# Seems like increasing the square footage yield the best returns
###