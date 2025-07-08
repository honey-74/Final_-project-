# Final_-project-
 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data = {
    'Device': ['Smartphone', 'Laptop', 'Refrigerator', 'Television', 'Microwave', 'Printer', 'Tablet', 'Washing Machine'],
    'Weight_kg': [0.2, 2.5, 60, 15, 12, 8, 0.5, 70],
    'Power_Consumption': [5, 45, 200, 120, 80, 30, 10, 250],
    'Type_Label': ['Consumer Electronics', 'IT Equipment', 'Large Appliances', 'Consumer Electronics',
                   'Large Appliances', 'IT Equipment', 'IT Equipment', 'Large Appliances']
}

df = pd.DataFrame(data)

df['Label'] = df['Type_Label'].astype('category').cat.codes
label_mapping = dict(enumerate(df['Type_Label'].astype('category').cat.categories))

X = df[['Weight_kg', 'Power_Consumption']]
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_mapping.values()))

def classify_e_waste(weight, power):
    category = clf.predict([[weight, power]])[0]
    return label_mapping[category]

print("\nPrediction Example:")
device_type = classify_e_waste(1.0, 40)
print(f"Device (1kg, 40W) classified as: {device_type}")
