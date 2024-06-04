import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from lib.create_grid import createGrid
from lib.block_bfs_signal_propagation import find_best_block

floor_plans = []
optimal_positions = []
block_size = 10

for i in range(10) :
  for j in range(1, 30) :
    grid = createGrid(str(i) + '_' +  str(j) + '.jpg', base='floor_plans/images/')
    best_block, best_coverage = find_best_block(grid, block_size)
    floor_plans.append(grid)
    optimal_positions.append(best_block)

for i in range(13, 23) :
  for j in range(1, 30) :
    grid = createGrid(str(i) + '_' +  str(j) + '.jpg', base='floor_plans/images/')
    best_block, best_coverage = find_best_block(grid, block_size)
    floor_plans.append(grid)
    optimal_positions.append(best_block)

for i in range(30, 36) :
  for j in range(1, 30) :
    grid = createGrid(str(i) + '_' +  str(j) + '.jpg', base='floor_plans/images/')
    best_block, best_coverage = find_best_block(grid, block_size)
    floor_plans.append(grid)
    optimal_positions.append(best_block)

for i in range(39, 45) :
  for j in range(1, 30) :
    grid = createGrid(str(i) + '_' +  str(j) + '.jpg', base='floor_plans/images/')
    best_block, best_coverage = find_best_block(grid, block_size)
    floor_plans.append(grid)
    optimal_positions.append(best_block)

for j in range(0, 100) :
  grid = createGrid(str(j) + '.png', base='simulations/images/')
  best_block, best_coverage = find_best_block(grid, block_size)
  floor_plans.append(grid)
  optimal_positions.append(best_block)

# Prepare data
X = np.array([fp.flatten() for fp in floor_plans])
M, N = floor_plans[0].shape
y = np.array([i * N + j for (i, j) in optimal_positions])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(M * N,)),
    Dense(64, activation='relu'),
    Dense(M * N, activation='softmax')  # Output layer with M * N units
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

model.save('mlp_model.h5')