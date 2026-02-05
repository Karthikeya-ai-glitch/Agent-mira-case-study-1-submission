import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# LOAD DATA
df = pd.read_excel("Case Study 1 Data.xlsx")

print("Dataset Shape:", df.shape)
display(df.head())

# BASIC INFO
df.info()

# MISSING VALUE SUMMARY
missing_summary = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage (%)": (df.isnull().sum() / len(df) * 100).round(2).values,
    "Data_Type": df.dtypes.values
}).sort_values(by="Missing_Percentage (%)", ascending=False)

display(missing_summary)

# SKY BLUE THEME
plt.rcParams.update({
    "axes.facecolor": "#EAF6FF",
    "figure.facecolor": "#EAF6FF",
    "axes.edgecolor": "#5DADE2",
    "axes.labelcolor": "#1B4F72",
    "xtick.color": "#1B4F72",
    "ytick.color": "#1B4F72",
    "text.color": "#1B4F72"
})

# TARGET DISTRIBUTION
plt.figure(figsize=(8,5))
plt.hist(df["Price"], bins=30, edgecolor="#1B4F72")
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# NUMERICAL FEATURE ANALYSIS
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    if col != "Price":
        plt.figure(figsize=(6,4))
        plt.scatter(df[col], df["Price"], alpha=0.6)
        plt.xlabel(col)
        plt.ylabel("Price")
        plt.title(f"{col} vs Price")
        plt.show()

# CATEGORICAL FEATURE ANALYSIS
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    plt.figure(figsize=(7,4))
    df.groupby(col)["Price"].mean().sort_values().plot(kind="bar")
    plt.title(f"Average Price by {col}")
    plt.ylabel("Average Price")
    plt.show()

# KNN IMPUTATION
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_encoded = df.copy()
df_encoded[cat_cols] = encoder.fit_transform(df_encoded[cat_cols])

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_encoded),
    columns=df_encoded.columns
)

print("Missing values after imputation:")
print(df_imputed.isna().sum())

# CORRELATION HEATMAP
plt.figure(figsize=(10,6))
sns.heatmap(df_imputed.corr(), cmap="Blues", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# FEATURE ENGINEERING FROM DATE SOLD
df_imputed["Date Sold"] = pd.to_datetime(df_imputed["Date Sold"])
df_imputed["Sold_Year"] = df_imputed["Date Sold"].dt.year
df_imputed["Sold_Month"] = df_imputed["Date Sold"].dt.month
df_imputed["Sold_Quarter"] = df_imputed["Date Sold"].dt.quarter
df_imputed["Property_Age_At_Sale"] = df_imputed["Sold_Year"] - df_imputed["Year Built"]

df_imputed = df_imputed.drop(columns=["Date Sold"])

display(df_imputed.head())
print("Final Dataset Shape:", df_imputed.shape)

# BEFORE VS AFTER SCALING
num_cols_scaled = df_imputed.select_dtypes(include=["int64", "float64"]).columns
num_cols_scaled = [col for col in num_cols_scaled if col != "Price"]

df_num = df_imputed[num_cols_scaled]

scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_num),
    columns=num_cols_scaled
)

fig, axes = plt.subplots(1, 2, figsize=(14,5))

axes[0].boxplot(df_num.values, labels=df_num.columns)
axes[0].set_title("Before Standardization")
axes[0].tick_params(axis="x", rotation=45)

axes[1].boxplot(df_scaled.values, labels=df_scaled.columns)
axes[1].set_title("After Standardization")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
