import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import pickle

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
df = pd.read_csv("fraud.csv")
print("Dataset loaded successfully")

# -----------------------------
# 2️⃣ Check Class Distribution
# -----------------------------
print("Original Class Distribution:")
print(df['Class'].value_counts())

# -----------------------------
# 3️⃣ Balance the Dataset
# -----------------------------
df_majority = df[df.Class == 0]
df_minority = df[df.Class == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

print("Balanced Class Distribution:")
print(df_balanced['Class'].value_counts())

# -----------------------------
# 4️⃣ Select Features and Target (Only Time & Amount for demo)
# -----------------------------
X = df_balanced[['Time', 'Amount']]  # Only 2 features
y = df_balanced['Class']

# -----------------------------
# 5️⃣ Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6️⃣ Train Model
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# 7️⃣ Save Model
# -----------------------------
pickle.dump(model, open("fraud_model.pkl", "wb"))  # Matches app.py
print("Model trained and saved successfully")
