# Student Score Predictor - Simple Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create dataset manually
data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62]
}
df = pd.DataFrame(data)

# Separate features and labels
X = df[['Hours']]
y = df['Scores']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict score for 9.25 hours
predicted_score = model.predict([[9.25]])
print(f"Predicted Score for 9.25 hours of study: {predicted_score[0]:.2f}")

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.show()
