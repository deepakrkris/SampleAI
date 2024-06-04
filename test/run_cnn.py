import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import json

grid_files = ['grid_rows_4.json', 'grid_rows_5.json', 'grid_rows_6.json', 'grid_rows_7.json']

floor_plans = []

for file in grid_files :
    with open(file, 'r') as f:
      grid_from_file = json.load(f)
      floor_plans.append(np.array(grid_from_file))  

optimal_positions = [
    (160, 80),
    (90, 90),
    (90, 90),
    (150, 70),
]

print(len(floor_plans))

# Prepare data
X = np.array(floor_plans)
M, N = X.shape[1], X.shape[2]
y = np.array([i * N + j for (i, j) in optimal_positions])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(M, N, 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(M * N, activation='softmax')  # Output layer with M * N units
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape data to add channel dimension
X_train = X_train.reshape(-1, M, N, 1)
X_test = X_test.reshape(-1, M, N, 1)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')
