import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

# Select useful columns
df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]

# Fill missing age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Convert gender into numbers
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])

# Features and target
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("TITANIC SURVIVAL PREDICTION")
print("---------------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))