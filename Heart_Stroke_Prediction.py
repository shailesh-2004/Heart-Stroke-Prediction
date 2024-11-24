import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_csv("C:\Users\OneDrive\Desktop\shailesh\Data_set.csv")
data.isnull().sum()

data = data.dropna(subset=['Symptoms'])

# Extraction of HDL values from 'Cholesterol Levels' column and convert into integers
data['HDL'] = data['Cholesterol Levels'].str.extract(r'HDL: (\d+)').astype(int)

# Extraction of LDL values from 'Cholesterol Levels' column and convert into integers
data['LDL'] = data['Cholesterol Levels'].str.extract(r'LDL: (\d+)').astype(int)

# Split the 'Blood Pressure Levels' column at '/' and create two new columns : 'Systolic_BP' & 'Diastolic_BP'
data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure Levels'].str.split('/', expand=True).astype(int)
data=data.drop(['Patient ID','Patient Name','Blood Pressure Levels','Cholesterol Levels'],axis=1)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
categoricalcolumns = ['Gender', 'Marital Status', 'Work Type','Residence Type','Smoking Status','Alcohol Intake','Physical Activity','Family History of Stroke','Dietary Habits','Diagnosis']

# Apply label encoding to each categorical column
for column in categoricalcolumns:
    data[column] = labelencoder.fit_transform(data[column])

# convert the 'Symptoms' column to string data type
data['Symptoms'] = data['Symptoms'].astype(str)
import spacy

#Loading English language model "en_core_web_sm"
nlp=spacy.load("en_core_web_sm")
def preprocess(text):
    doc = nlp(text)
    list = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        list.append(token.lemma_)
    return " ".join(list)

data['preprocessed_Symptoms'] = data['Symptoms'].apply(preprocess)

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
tfidf_vectorizer  = TfidfVectorizer()

# Apply the TF-IDF vectorizer to 'preprocessed_Symptoms' column
tfidf_matrix  = tfidf_vectorizer.fit_transform(data['preprocessed_Symptoms'])

# Convert the TF-IDF matrix to a DataFrame
tfidf_data = pd.DataFrame(tfidf_matrix .toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Drop the columns 'Symptoms' and 'preprocessed_Symptoms'
data=data.drop(['Symptoms','preprocessed_Symptoms'],axis=1)

# Combine the TF-IDF DataFrame with the original DataFrame
new_data = pd.concat([data, tfidf_data], axis=1)
new_data.dropna(inplace=True)

# Split the data into features and target
X = new_data.drop('Diagnosis', axis=1)
y = new_data['Diagnosis']

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=90,random_state=45)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
