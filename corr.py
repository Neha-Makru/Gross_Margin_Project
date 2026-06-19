import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("worker.xlsx", sheet_name="ml_training_dataset")

# Identify all numerical columns automatically
numerical_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_matrix = numerical_df.corr()

# Focus specifically on how everything relates to "Gross Margin %"
target_corr = corr_matrix["target_actual_gm_pct"].sort_values(ascending=False)

print("Correlation with Gross Margin % (All Numerical Columns):")
print("-" * 50)
print(target_corr)


# # Save the full matrix to a CSV for detailed review
# corr_matrix.to_csv("full_correlation_matrix.csv")
# print("\nFull correlation matrix saved to 'full_correlation_matrix.csv'")