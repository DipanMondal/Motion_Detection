import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the filtered dataset
data = pd.read_csv(r"data/skeleton_data.csv")

# Separate features and labels
X = data.drop('label',axis=1).values  # All columns except the last one (features)
y = data['label']   # Last column (labels)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

# Standardize the features for better performance of SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for use in real-time detection
joblib.dump(scaler, r"models/scaler.pkl")
print("Scaler saved to scaler.pkl")

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)  # Using linear kernel
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the trained model for future use
joblib.dump(svm_model, r"models/svm_motion_model1.pkl")
print("SVM model saved to svm_motion_model1.pkl")
