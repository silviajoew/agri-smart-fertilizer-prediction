import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. LOAD THE DATA
df = pd.read_csv('fertilizer_recommendation.csv')
print(f"Loaded {len(df)} rows.")

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# 2. AUTOMATIC ENCODING FOR ALL TEXT COLUMNS
encoders = {}
text_columns = df.select_dtypes(include=['object']).columns

for col in text_columns:
    print(f"Encoding text column: {col}")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 3. SET THE TARGET
target_col = 'recommended_fertilizer'

# 4. SPLIT DATA
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TRAIN THE MODEL
print(f"Training Random Forest on {len(X_train)} rows...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. TEST ACCURACY
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%")

# --- 7. NEW: GENERATE CONFUSION MATRIX CHART ---
# This part creates the visual chart for your report
cm = confusion_matrix(y_test, y_pred)
labels = encoders[target_col].classes_ # Get original fertilizer names

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=labels, yticklabels=labels)

plt.title('Confusion Matrix - AgriSmart Fertilizer Prediction')
plt.xlabel('Predicted Fertilizer')
plt.ylabel('Actual Fertilizer')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Save the chart as an image for your screenshot
plt.savefig('confusion_matrix_chart.png')
print("\nSuccess! Confusion Matrix saved as 'confusion_matrix_chart.png'")
# plt.show() # Uncomment this if you are running in a local IDE to see it pop up

# 8. SAVE THE BRAIN
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model, 
        'encoders': encoders,
        'feature_columns': X.columns.tolist(),
        'target_col': target_col
    }, f)

print("Project files updated successfully!")