import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


file_path = 'xssdata.xlsx'
df = pd.read_excel(file_path)


print("First few rows of the dataset:")
print(df.head())
print("Column names in the dataset:")
print(df.columns)


df['Label'] = df['Attack Type'].apply(lambda x: 0 if x == "Normal" else 1)


print("Number of benign samples:", len(df[df['Label'] == 0]))
print("Number of malicious samples:", len(df[df['Label'] == 1]))


df_balanced = df.copy()


df_balanced = df_balanced.dropna(subset=['Payload'])
df_balanced = df_balanced[df_balanced['Payload'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]


print("Number of NaN values in 'Payload' column after cleaning:", df_balanced['Payload'].isna().sum())


df_balanced.to_excel('cleaned_xssdata.xlsx', index=False)
print("Cleaned dataset saved to 'cleaned_xssdata.xlsx'.")


vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 3))
X_train = vectorizer.fit_transform(df_balanced['Payload'])


model = IsolationForest(contamination=0.3, random_state=42)
model.fit(X_train)


joblib.dump(model, 'isolation_forest_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Isolation Forest model and vectorizer trained successfully.")
print("Model saved to isolation_forest_model.pkl and vectorizer saved to vectorizer.pkl.")

