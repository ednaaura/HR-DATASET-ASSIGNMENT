# HR-DATASET-ASSIGNMENT
## HR dataset PCA assigment 
## HUMAN RESOURCE ASSIGNMENT

## PART A:Basic EDA

### 1.Load the dataset into a Pandas DataFrame and display the first 5 rows.
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#import dataset
df=pd.read_csv("C:/Users/admin/Desktop/LUXTECH/MACHINE LEARNING/HRDataset_v14.csv")
df.head(10)
# 2.Check the shape of the dataset (rows, columns)
print (df.shape)
- dataset has 311 rows and 36 columns.
#3.Display the column names and their data types
df.dtypes
#4.check number of unique values
df.nunique()
#5.check for missing values in the dataset
df.isnull().sum()
- dateoftermination had missing values of 207.
#6.describe numerical columns
df.describe()
#7.plot salary distribution
plt.figure(figsize=(10, 6)) 
plt.hist(df['Salary'], bins=30, alpha=0.7, color='skyblue')
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()
#8.What is the average age of employees at the company? (Use DOB column).
df['DOB'] = pd.to_datetime(df['DOB'])
current_year = datetime.now().year - df['DOB'].dt.year
average_age = current_year.mean()
print(average_age)
#9.How many employees are still employed vs terminated?
employment_status = df['EmploymentStatus'].value_counts()
print(employment_status)
- terminated employees are 104(both voluntary and involuntary) while employed actively are 207.
#10.Which departments have the most employees?
#plotting a countplot
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Department', order=df['Department'].value_counts().index) 
plt.title('Number of Employees per Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.show()
- from the visualization, production department had the most  employees with approximately 220.
## PART B: Business Analysis

#11.What is the average salary per department?
avg_salary_dept= df.groupby('Department')['Salary'].mean().sort_values(ascending=False)
print(avg_salary_dept)
#12.employment status distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='EmploymentStatus', palette='Set2')
plt.title('Employment Status Distribution') 
plt.xlabel('Employment Status')
plt.ylabel('Count')
plt.show()

#13.employment status distribution using pie chart
employment_counts = df['EmploymentStatus'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(employment_counts, labels= employment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set3'))
plt.title('Employment Status Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#14.Compare salary levels between Male and Female employees.
#using boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df,x='Sex',y='Salary')
plt.title('Salary Distribution by Gender')
plt.show()
#15.Which recruitment source brings in the most employees?
recruitment_sources=df['RecruitmentSource'].value_counts()
print(recruitment_sources)
- it is shown that 'Indeed' brings in the most employees.
#16.What percentage of employees attended a Diversity Job Fair?
diversity_percentage=df['FromDiversityJobFairID'].mean()*100
print(diversity_percentage)
#17.Compare engagement survey scores across different departments.
#plotting barplot
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Department', y='EngagementSurvey', ci=None, palette='viridis')
plt.title('Average Engagement Survey Scores by Department')
plt.xlabel('Department')
plt.ylabel('Average Engagement Survey Score')
plt.tight_layout()
plt.show()
- executive office department had the highest engagement score while sales departmnt had the lowest engagement score.
#18.Which race has the highest average salary?
avg_salary_race= df.groupby('RaceDesc')['Salary'].mean().sort_values(ascending=False)
print(avg_salary_race)
#19.What is the relationship between number of projects (SpecialProjectsCount) and salary?
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='SpecialProjectsCount', y='Salary', hue='Department', palette='tab10', alpha=0.7)
plt.title('Special Projects Count vs Salary')       
plt.show()
#20.Do married employees earn more on average than single employees?
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='MaritalDesc', y='Salary')
plt.title('Average Salary by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Average Salary')
plt.show()
#21.Which managers have the largest teams?
manager_teams=df.groupby('ManagerName').size().sort_values(ascending=False)
print(manager_teams)
## PART C: Data Visualization
#22.Plot the salary distribution using histograms.
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.hist(df['Salary'],bins=30,alpha=0.7,color='purple')
plt.title('Salary Distribution')
plt.show()
#23.Show the count of employees by department
plt.subplot(2,2,2)
df['Department'].value_counts().plot(kind='bar')
plt.title('Employee Count by Department')
plt.show()
#24.Compare average satisfaction score by department.
plt.subplot(2,2,3)
df.groupby('Department')['EmpSatisfaction'].mean().plot(kind='bar')
plt.title("Average Satisfaction by Department")
plt.show()

#25.Visualize employee terminations over time.
plt.subplot(2, 2, 4)
terminated = df[df['Termd'] == 1]
if not terminated.empty:
    terminated['DateofTermination'] = pd.to_datetime(terminated['DateofTermination'])
    terminated.groupby(terminated['DateofTermination'].dt.year).size().plot()
    plt.title('Terminations Over Time')
else:
    plt.text(0.5, 0.5, 'No termination data', ha='center')

plt.tight_layout()
plt.show()
#Plot average salary by gender using a boxplot.
plt.figure(figsize=(10,6))
sns.boxplot(data=df,x='Sex',y='Salary')
plt.title('Salary Distribution by Gender')
plt.show()

#26.Visualize performance scores vs salary.
plt.figure(figsize=(12, 8))
sns.stripplot(data=df, x='PerformanceScore', y='Salary', jitter=True)
plt.title('Performance Score vs Salary')
plt.show()
#27.Create a heatmap of correlations between numeric variables.
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
#28.Plot engagement survey score vs satisfaction score.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='EngagementSurvey', y='EmpSatisfaction')
plt.title('Engagement vs Satisfaction')
plt.show()
#29.Show a stacked bar chart of employee status across departments. 
status_dept = pd.crosstab(df['Department'], df['EmploymentStatus'])
status_dept.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Employment Status by Department')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
#30.Plot absenteeism (Absences) distribution among employees.
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Absences', bins=20)
plt.title('Absenteeism Distribution')
plt.show()
## PART D : PCA (Dimensionality Reduction)
#31.Standardize numerical features before applying PCA.
numeric_df = df.select_dtypes(include=[np.number]).dropna()
#32.Perform PCA on the dataset and explain the first 2 components.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)
#33.Plot the explained variance ratio for the PCA components.
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

#
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance Ratio')
plt.grid(True)
plt.show()
#34.Reduce the dataset to 2 dimensions with PCA and plot employees colored by department.
pca_2d = PCA(n_components=2)
pca_result_2d = pca_2d.fit_transform(scaled_data)

#35.Which variables contribute most to the first principal component?
component_df = pd.DataFrame({
    'Feature': numeric_df.columns,
    'PC1': pca.components_[0],
    'PC2': pca.components_[1]
})
print("Top contributors to first principal component:")
print(component_df.nlargest(5, 'PC1')[['Feature', 'PC1']])
print("\nTop contributors to second principal component:")
print(component_df.nlargest(5, 'PC2')[['Feature', 'PC2']])

#36.Apply PCA to engagement + satisfaction + absences. Do these condense into 1 dimension?
selected_features = ['EngagementSurvey', 'EmpSatisfaction', 'Absences']
selected_df = df[selected_features].dropna()
scaler_selected = StandardScaler()
scaled_selected = scaler_selected.fit_transform(selected_df)

pca_selected = PCA()
pca_selected_result = pca_selected.fit_transform(scaled_selected)
print(f"\nVariance explained by first component for selected features: {pca_selected.explained_variance_ratio_[0]:.3f}")

# K-means clustering comparison
kmeans_original = KMeans(n_clusters=3, random_state=42)
kmeans_pca = KMeans(n_clusters=3, random_state=42)

original_clusters = kmeans_original.fit_predict(scaled_data)
pca_clusters = kmeans_pca.fit_predict(pca_result_2d)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=original_clusters)
plt.title('K-means Clustering (Original Data)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=pca_result_2d[:, 0], y=pca_result_2d[:, 1], hue=pca_clusters)
plt.title('K-means Clustering (PCA Reduced)')

plt.tight_layout()
plt.show()
# PCA loadings plot
plt.figure(figsize=(10, 8))
for i, feature in enumerate(['Salary', 'Absences', 'EngagementSurvey']):
    if feature in numeric_df.columns:
        idx = list(numeric_df.columns).index(feature)
        plt.arrow(0, 0, pca.components_[0, idx], pca.components_[1, idx], 
                 color='r', alpha=0.5)
        plt.text(pca.components_[0, idx]*1.15, pca.components_[1, idx]*1.15, 
                feature, color='r', ha='center', va='center')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Loadings Plot')
plt.grid(True)
plt.show()
