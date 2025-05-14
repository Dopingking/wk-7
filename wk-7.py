# 📦 Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 📌 Set plot style
sns.set(style="whitegrid")

try:
    # 🚀 Task 1: Load and Explore the Dataset
    iris_data = load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

    # ✅ Display first few rows
    print("🔍 First 5 rows of the dataset:")
    print(df.head())

    # ✅ Structure and missing values
    print("\n📋 Dataset Info:")
    print(df.info())

    print("\n❓ Missing Values Check:")
    print(df.isnull().sum())

    # 🚀 Task 2: Basic Data Analysis
    print("\n📊 Descriptive Statistics:")
    print(df.describe())

    print("\n📊 Mean values per species:")
    print(df.groupby('species').mean())

    print("\n📌 Observation:")
    print("• Setosa has smaller petal size than others.\n"
          "• Virginica has the highest average measurements.\n"
          "• Some features like petal length vary greatly by species.")

    # 🚀 Task 3: Data Visualization

    # 1️⃣ Line Chart – Simulated Time-Series
    df['index'] = range(len(df))
    plt.figure(figsize=(10, 5))
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.plot(subset['index'], subset['sepal length (cm)'], label=species)
    plt.title('Sepal Length Over Index (Simulated Time)')
    plt.xlabel('Index')
    plt.ylabel('Sepal Length (cm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2️⃣ Bar Chart – Mean Petal Length per Species
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='species', y='petal length (cm)', estimator='mean')
    plt.title('Average Petal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.tight_layout()
    plt.show()

    # 3️⃣ Histogram – Sepal Width Distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='sepal width (cm)', kde=True, bins=15, color='skyblue')
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.tight_layout()
    plt.show()

    # 4️⃣ Scatter Plot – Sepal vs. Petal Length
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
    plt.title('Sepal Length vs. Petal Length by Species')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.tight_layout()
    plt.show()

    # 🚀 Summary
    print("\n🧠 Summary of Insights:")
    print("- Setosa flowers are clearly different in petal size.")
    print("- Virginica tends to dominate in terms of size.")
    print("- Sepal length and petal length show a positive correlation.")
    print("- Histogram reveals the distribution is fairly normal for sepal width.")

except Exception as e:
    print(f"❌ Error during processing: {e}")
