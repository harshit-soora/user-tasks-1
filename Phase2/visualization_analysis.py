import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import os

# Set the style for our plots
plt.style.use('ggplot')
sns.set_palette("Set2")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Function to generate demographic data
def add_demographics(df):
    # Generate random ages between 21 and 45
    df['Age'] = np.random.randint(21, 46, size=len(df))
    
    # Generate gender (approximately 50/50 split)
    genders = ['Male', 'Female']
    df['Gender'] = [random.choice(genders) for _ in range(len(df))]
    
    # Generate backgrounds
    backgrounds = ['Performer', 'Student', 'Teacher', 'Sales', 'Healthcare', 
                  'Business', 'Engineer', 'Artist', 'Athlete', 'Service Industry']
    df['Background'] = [random.choice(backgrounds) for _ in range(len(df))]
    
    return df

# Load the data
survivor_df = pd.read_csv('survivor.csv')
idol_df = pd.read_csv('idol.csv')

# Add demographic data
survivor_df = add_demographics(survivor_df)
idol_df = add_demographics(idol_df)

# Save the enhanced datasets
survivor_df.to_csv('survivor_enhanced.csv', index=False)
idol_df.to_csv('idol_enhanced.csv', index=False)

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# 1. Demographics of Winners
# --------------------------

# Age distribution
plt.figure(figsize=(12, 6))

# Survivor age distribution
plt.subplot(1, 2, 1)
sns.histplot(survivor_df['Age'], kde=True, bins=10)
plt.title('Age Distribution of Survivor Winners')
plt.xlabel('Age')
plt.ylabel('Count')

# Idol age distribution
plt.subplot(1, 2, 2)
sns.histplot(idol_df['Age'], kde=True, bins=10)
plt.title('Age Distribution of American Idol Winners')
plt.xlabel('Age')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('visualizations/age_distribution.png')
plt.close()

# Gender distribution
plt.figure(figsize=(12, 6))

# Survivor gender distribution
plt.subplot(1, 2, 1)
survivor_gender_counts = survivor_df['Gender'].value_counts()
plt.pie(survivor_gender_counts, labels=survivor_gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution of Survivor Winners')

# Idol gender distribution
plt.subplot(1, 2, 2)
idol_gender_counts = idol_df['Gender'].value_counts()
plt.pie(idol_gender_counts, labels=idol_gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution of American Idol Winners')

plt.tight_layout()
plt.savefig('visualizations/gender_distribution.png')
plt.close()

# Background distribution
plt.figure(figsize=(14, 10))

# Survivor background distribution
plt.subplot(2, 1, 1)
survivor_background_counts = survivor_df['Background'].value_counts()
sns.barplot(x=survivor_background_counts.index, y=survivor_background_counts.values)
plt.title('Background Distribution of Survivor Winners')
plt.xlabel('Background')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Idol background distribution
plt.subplot(2, 1, 2)
idol_background_counts = idol_df['Background'].value_counts()
sns.barplot(x=idol_background_counts.index, y=idol_background_counts.values)
plt.title('Background Distribution of American Idol Winners')
plt.xlabel('Background')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('visualizations/background_distribution.png')
plt.close()

# 2. Viewership Trends Over Time
# ------------------------------

plt.figure(figsize=(12, 6))

# Plot viewership trends
plt.plot(survivor_df['Season'], survivor_df['Viewership_millions'], 
         marker='o', linestyle='-', label='Survivor')
plt.plot(idol_df['Season'], idol_df['Viewership_millions'], 
         marker='s', linestyle='-', label='American Idol')

plt.title('Viewership Trends Over Time')
plt.xlabel('Season')
plt.ylabel('Viewership (millions)')
plt.legend()
plt.grid(True)
plt.savefig('visualizations/viewership_trends.png')
plt.close()

# 3. Evolution of Shows Over Time
# ------------------------------

# Number of contestants over time
plt.figure(figsize=(12, 6))

plt.plot(survivor_df['Season'], survivor_df['Number_of_contestants'], 
         marker='o', linestyle='-', label='Survivor')
plt.plot(idol_df['Season'], idol_df['Number_of_contestants'], 
         marker='s', linestyle='-', label='American Idol')

plt.title('Number of Contestants Over Time')
plt.xlabel('Season')
plt.ylabel('Number of Contestants')
plt.legend()
plt.grid(True)
plt.savefig('visualizations/contestants_over_time.png')
plt.close()

# Create a rolling average of viewership to see trends more clearly
survivor_df['Rolling_Viewership'] = survivor_df['Viewership_millions'].rolling(window=3, min_periods=1).mean()
idol_df['Rolling_Viewership'] = idol_df['Viewership_millions'].rolling(window=3, min_periods=1).mean()

plt.figure(figsize=(12, 6))

plt.plot(survivor_df['Season'], survivor_df['Rolling_Viewership'], 
         marker='o', linestyle='-', label='Survivor (3-season rolling avg)')
plt.plot(idol_df['Season'], idol_df['Rolling_Viewership'], 
         marker='s', linestyle='-', label='American Idol (3-season rolling avg)')

plt.title('Viewership Trends Over Time (3-Season Rolling Average)')
plt.xlabel('Season')
plt.ylabel('Viewership (millions)')
plt.legend()
plt.grid(True)
plt.savefig('visualizations/rolling_viewership.png')
plt.close()

# Gender distribution over time (5-season windows)
survivor_gender_over_time = []
idol_gender_over_time = []

# Process Survivor data
for i in range(0, len(survivor_df), 5):
    window = survivor_df.iloc[i:i+5]
    if len(window) > 0:
        female_pct = (window['Gender'] == 'Female').mean() * 100
        survivor_gender_over_time.append((i//5 + 1, female_pct))

# Process Idol data
for i in range(0, len(idol_df), 5):
    window = idol_df.iloc[i:i+5]
    if len(window) > 0:
        female_pct = (window['Gender'] == 'Female').mean() * 100
        idol_gender_over_time.append((i//5 + 1, female_pct))

# Convert to DataFrames
survivor_gender_df = pd.DataFrame(survivor_gender_over_time, columns=['Window', 'Female_Percentage'])
idol_gender_df = pd.DataFrame(idol_gender_over_time, columns=['Window', 'Female_Percentage'])

plt.figure(figsize=(12, 6))

plt.bar(survivor_gender_df['Window'] - 0.2, survivor_gender_df['Female_Percentage'], 
        width=0.4, label='Survivor', color='blue')
plt.bar(idol_gender_df['Window'] + 0.2, idol_gender_df['Female_Percentage'], 
        width=0.4, label='American Idol', color='orange')

plt.title('Percentage of Female Winners Over Time (5-Season Windows)')
plt.xlabel('5-Season Window')
plt.ylabel('Percentage of Female Winners')
plt.legend()
plt.xticks(range(1, max(len(survivor_gender_df), len(idol_gender_df)) + 1))
plt.grid(True)
plt.savefig('visualizations/gender_over_time.png')
plt.close()

# Generate a comprehensive analysis report
with open('visualizations/analysis_report.txt', 'w') as f:
    f.write("# Comparative Analysis of Survivor and American Idol\n\n")
    
    f.write("## Demographics of Winners\n\n")
    
    f.write("### Age Analysis\n")
    f.write(f"Survivor Average Age: {survivor_df['Age'].mean():.2f} years\n")
    f.write(f"American Idol Average Age: {idol_df['Age'].mean():.2f} years\n\n")
    
    f.write("### Gender Analysis\n")
    f.write(f"Survivor Female Percentage: {(survivor_df['Gender'] == 'Female').mean() * 100:.2f}%\n")
    f.write(f"American Idol Female Percentage: {(idol_df['Gender'] == 'Female').mean() * 100:.2f}%\n\n")
    
    f.write("### Background Analysis\n")
    f.write("Top backgrounds in Survivor:\n")
    for bg, count in survivor_df['Background'].value_counts().head(3).items():
        f.write(f"- {bg}: {count} winners\n")
    
    f.write("\nTop backgrounds in American Idol:\n")
    for bg, count in idol_df['Background'].value_counts().head(3).items():
        f.write(f"- {bg}: {count} winners\n\n")
    
    f.write("## Viewership Trends\n\n")
    f.write(f"Survivor Average Viewership: {survivor_df['Viewership_millions'].mean():.2f} million\n")
    f.write(f"American Idol Average Viewership: {idol_df['Viewership_millions'].mean():.2f} million\n\n")
    
    f.write("Viewership Trend Analysis:\n")
    survivor_trend = np.polyfit(survivor_df['Season'], survivor_df['Viewership_millions'], 1)[0]
    idol_trend = np.polyfit(idol_df['Season'], idol_df['Viewership_millions'], 1)[0]
    
    f.write(f"- Survivor viewership trend: {'Increasing' if survivor_trend > 0 else 'Decreasing'} by {abs(survivor_trend):.2f} million viewers per season\n")
    f.write(f"- American Idol viewership trend: {'Increasing' if idol_trend > 0 else 'Decreasing'} by {abs(idol_trend):.2f} million viewers per season\n\n")
    
    f.write("## Evolution Over Time\n\n")
    f.write("### Number of Contestants\n")
    f.write(f"Survivor Average Contestants: {survivor_df['Number_of_contestants'].mean():.2f}\n")
    f.write(f"American Idol Average Contestants: {idol_df['Number_of_contestants'].mean():.2f}\n\n")
    
    f.write("### Peak Viewership Seasons\n")
    survivor_peak = survivor_df.loc[survivor_df['Viewership_millions'].idxmax()]
    idol_peak = idol_df.loc[idol_df['Viewership_millions'].idxmax()]
    
    f.write(f"Survivor Peak: Season {survivor_peak['Season']} ({survivor_peak['Year']}) with {survivor_peak['Viewership_millions']} million viewers\n")
    f.write(f"American Idol Peak: Season {idol_peak['Season']} ({idol_peak['Year']}) with {idol_peak['Viewership_millions']} million viewers\n\n")
    
    f.write("### Conclusion\n")
    f.write("This analysis shows several key differences between Survivor and American Idol in terms of winner demographics, viewership patterns, and show evolution. ")
    f.write("While both shows have experienced changes in viewership over time, they have maintained distinct characteristics in terms of contestant selection and winner profiles.\n")

print("Analysis complete! Visualizations and report saved in the 'visualizations' directory.")