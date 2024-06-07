from flask import Flask, request, jsonify, send_file
from lib.create_grid import createGrid , createGrid_m2 , createGrid_m3
from lib.block_bfs_signal_propagation import find_best_block
from lib.generate_ap_location_image import grid_to_image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved models
mlp_model = load_model('ml_models/mlp_model.h5')
cnn_model = load_model('ml_models/cnn_model.h5')
mlp_model_2 = load_model('ml_models/mlp_model_2.h5')
cnn_model_2 = load_model('ml_models/cnn_model_2.h5')

@app.route('/get_image/<file_name>', methods=['GET'])
def get_image(file_name):
    return send_file('floor_plans/images/' + file_name, mimetype='image/gif')

@app.route('/predict/m1/<file_name>', methods=['GET'])
def get_best_locations(file_name):
    grid = createGrid(file_name)

    block_size = 10
    best_block, best_coverage = find_best_block(grid, block_size)

    return [best_block, best_coverage]

@app.route('/topo/m1/<file_name>', methods=['GET'])
def get_topology(file_name):
    grid = createGrid(file_name)

    block_size = 10
    best_block, best_coverage = find_best_block(grid, block_size)

    print(best_block, best_coverage)

    image_bytes = grid_to_image(grid, colormap="plasma", scale_factor=5, icon_position=(best_block[1], best_block[0]))

    return send_file(image_bytes, mimetype='image/gif')

@app.route('/predict/m2/<file_name>', methods=['GET'])
def get_best_locations_m2(file_name):
    grid = createGrid_m2(file_name)

    block_size = 10
    best_block, best_coverage = find_best_block(grid, block_size)

    return [best_block, best_coverage]

@app.route('/topo/m2/<file_name>', methods=['GET'])
def get_topology_m2(file_name):
    grid = createGrid_m2(file_name)

    block_size = 10
    best_block, best_coverage = find_best_block(grid, block_size)

    print(best_block, best_coverage)

    image_bytes = grid_to_image(grid, colormap="plasma", scale_factor=5, icon_position=(best_block[1], best_block[0]))

    return send_file(image_bytes, mimetype='image/gif')

@app.route('/topo/m3/<file_name>', methods=['GET'])
def get_topology_m3(file_name):
    grid = createGrid_m3(file_name)

    block_size = 20
    best_block, best_coverage = find_best_block(grid, block_size)

    print(best_block, best_coverage)

    image_bytes = grid_to_image(grid, colormap="plasma", scale_factor=5, icon_position=(best_block[1], best_block[0]))

    return send_file(image_bytes, mimetype='image/gif')

@app.route('/mlp_predict/m1/<file_name>', methods=['GET'])
def predict_position_mlp(file_name):
    grid = createGrid(file_name)

    # Predict optimal position for a new floor plan
    predicted_position = mlp_model.predict(grid.flatten().reshape(1, -1))
    print(predicted_position)
    predicted_index = np.argmax(predicted_position)
    i, j = divmod(predicted_index, 100)
    print(f'Predicted optimal position: ({i}, {j})')
    return [i.item(), j.item()]

@app.route('/cnn_predict/m1/<file_name>', methods=['GET'])
def predict_position_cnn(file_name):
    grid = createGrid(file_name)

    # Predict optimal position for a new floor plan
    predicted_position = cnn_model.predict(grid.flatten().reshape(1, 100, 100, 1))
    print(predicted_position)
    predicted_index = np.argmax(predicted_position)
    i, j = divmod(predicted_index, 100)
    print(f'Predicted optimal position: {i}, {j}')
    return [i.item(), j.item()]


@app.route('/mlp_topo/m1/<file_name>', methods=['GET'])
def mlp_topo(file_name):
    grid = createGrid(file_name)

    # Predict optimal position for a new floor plan
    predicted_position = mlp_model.predict(grid.flatten().reshape(1, -1))
    print(predicted_position)
    predicted_index = np.argmax(predicted_position)
    i, j = divmod(predicted_index, 100)
    print(f'Predicted optimal position: ({i}, {j})')

    image_bytes = grid_to_image(grid, colormap="plasma", scale_factor=5, icon_position=(j, i))

    return send_file(image_bytes, mimetype='image/gif')

@app.route('/cnn_topo/m1/<file_name>', methods=['GET'])
def cnn_topo(file_name):
    grid = createGrid(file_name)

    # Predict optimal position for a new floor plan
    predicted_position = cnn_model.predict(grid.flatten().reshape(1, 100, 100, 1))
    print(predicted_position)
    predicted_index = np.argmax(predicted_position)
    i, j = divmod(predicted_index, 100)
    print(f'Predicted optimal position: {i}, {j}')

    image_bytes = grid_to_image(grid, colormap="plasma", scale_factor=5, icon_position=(j, i))

    return send_file(image_bytes, mimetype='image/gif')


@app.route('/mlp_predict/m2/<file_name>', methods=['GET'])
def predict_position_mlp_m2(file_name):
    grid = createGrid_m2(file_name)

    # Predict optimal position for a new floor plan
    predicted_position = mlp_model_2.predict(grid.flatten().reshape(1, -1))
    print(predicted_position)
    predicted_index = np.argmax(predicted_position)
    i, j = divmod(predicted_index, 100)
    print(f'Predicted optimal position: ({i}, {j})')
    return [i.item(), j.item()]

@app.route('/cnn_predict/m2/<file_name>', methods=['GET'])
def predict_position_cnn_m2(file_name):
    grid = createGrid_m2(file_name)

    # Predict optimal position for a new floor plan
    predicted_position = cnn_model_2.predict(grid.flatten().reshape(1, 120, 120, 1))
    print(predicted_position)
    predicted_index = np.argmax(predicted_position)
    i, j = divmod(predicted_index, 100)
    print(f'Predicted optimal position: {i}, {j}')
    return [i.item(), j.item()]


@app.route('/mlp_topo/m2/<file_name>', methods=['GET'])
def mlp_topo_m2(file_name):
    grid = createGrid_m2(file_name)

    # Predict optimal position for a new floor plan
    predicted_position = mlp_model_2.predict(grid.flatten().reshape(1, -1))
    print(predicted_position)
    predicted_index = np.argmax(predicted_position)
    i, j = divmod(predicted_index, 100)
    print(f'Predicted optimal position: ({i}, {j})')

    image_bytes = grid_to_image(grid, colormap="plasma", scale_factor=5, icon_position=(j, i))

    return send_file(image_bytes, mimetype='image/gif')

@app.route('/cnn_topo/m2/<file_name>', methods=['GET'])
def cnn_topo_m2(file_name):
    grid = createGrid_m2(file_name)

    # Predict optimal position for a new floor plan
    predicted_position = cnn_model_2.predict(grid.flatten().reshape(1, 120, 120, 1))
    print(predicted_position)
    predicted_index = np.argmax(predicted_position)
    i, j = divmod(predicted_index, 100)
    print(f'Predicted optimal position: {i}, {j}')

    image_bytes = grid_to_image(grid, colormap="plasma", scale_factor=5, icon_position=(j, i))

    return send_file(image_bytes, mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)
