import cv2
import numpy as np

# Load floor plan image and convert to grayscale
image = cv2.imread("fp_grayscale.png", cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection (adjust parameters if needed)
edges = cv2.Canny(image, 50, 150)

# Invert the edge image (dark edges become white)
inverted_edges = 255 - edges

# Perform morphological operations (opening and closing)
kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size as needed
opened_image = cv2.morphologyEx(inverted_edges, cv2.MORPH_OPEN, kernel)
closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

# Target grid size
target_size = (100, 100)

# Calculate scaling factors for rows and columns
row_scale = closed_image.shape[0] / target_size[0]
col_scale = closed_image.shape[1] / target_size[1]

# Resample the processed image into a smaller grid with classification
small_grid = np.zeros(target_size)
for i in range(target_size[0]):
  for j in range(target_size[1]):
    # Get start and end indices for the corresponding region
    start_row = int(i * row_scale)
    end_row = int(min((i + 1) * row_scale, closed_image.shape[0]))
    start_col = int(j * col_scale)
    end_col = int(min((j + 1) * col_scale, closed_image.shape[1]))

    # Analyze presence in original binary image and processed image
    original_value = inverted_edges[start_row, start_col]
    processed_value = closed_image[start_row, start_col]

    # Assign values based on classification (adjust thresholds)
    if original_value == 255 and processed_value == 255:
        small_grid[i, j] = -10  # Wall (thick line)
    elif processed_value == 255:
        small_grid[i, j] = -5  # Door (thin line, thickened after closing)
    else:
        small_grid[i, j] = 0  # Open space

# Print the resulting 100x100 grid with wall, door, and open space markers
print(small_grid.astype(int))

grid = small_grid.astype(int)

with open('grid_rows_3.json', 'w') as f:
    f.write('[\n')
    for row in grid:
        row_str = ', '.join(f'{cell:3}' for cell in row)  # join elements with ', ' and avoid trailing comma
        f.write(f'  [{row_str}],\n')
    f.write(']\n')
