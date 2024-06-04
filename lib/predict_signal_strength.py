import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def simulate_data():
    # Simulate some data for training
    data = []
    for _ in range(10000):
        grid_size = 100
        grid = np.zeros((grid_size, grid_size))
        ap_x, ap_y = np.random.randint(0, grid_size, size=2)
        grid[ap_x, ap_y] = 10  # Mark the access point with initial strength
        signal_strength = np.zeros_like(grid)
        
        for i in range(grid_size):
            for j in range(grid_size):
                distance = np.abs(ap_x - i) + np.abs(ap_y - j)
                strength = 10 - distance
                signal_strength[i, j] = max(0, strength)
        
        # Adding walls randomly for the simulation
        num_walls = np.random.randint(1, 5)
        for _ in range(num_walls):
            wall_x, wall_y = np.random.randint(0, grid_size, size=2)
            grid[wall_x, wall_y] = -10  # Mark walls
        
        # Collect data points
        for i in range(grid_size):
            for j in range(grid_size):
                features = (i, j, ap_x, ap_y, grid[i, j])
                target = signal_strength[i, j]
                data.append((features, target))
    
    return data

def train():
    data = simulate_data()
    # Prepare the data
    X = np.array([x[0] for x in data])
    y = np.array([x[1] for x in data])

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a neural network regressor
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Training set score: %f" % model.score(X_train, y_train))
    print("Test set score: %f" % model.score(X_test, y_test))

    return scaler, model

def predict_signal_strength(scaler, model, grid, ap_x, ap_y):
    grid_size = grid.shape[0]
    signal_strength = np.zeros_like(grid)
    
    for i in range(grid_size):
        for j in range(grid_size):
            features = np.array([[i, j, ap_x, ap_y, grid[i, j]]])
            features = scaler.transform(features)
            strength = model.predict(features)
            signal_strength[i, j] = max(0, strength)
    
    return signal_strength

scaler, model = train()
