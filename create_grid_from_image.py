import cv2
import numpy as np

# Load floor plan image and convert to grayscale
image = cv2.imread("fp_grayscale.png", cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection (adjust parameters if needed)
edges = cv2.Canny(image, 50, 150)

# Invert the edge image (dark edges become white)
inverted_edges = edges

# Target grid size
target_size = (100, 100)

# Calculate scaling factors for rows and columns
row_scale = inverted_edges.shape[0] / target_size[0]
col_scale = inverted_edges.shape[1] / target_size[1]

# Resample the binary image into a smaller grid using maximum value
small_grid = np.zeros(target_size)
for i in range(target_size[0]):
  for j in range(target_size[1]):
    # Get start and end indices for the corresponding region
    start_row = int(i * row_scale)
    end_row = int(min((i + 1) * row_scale, inverted_edges.shape[0]))
    start_col = int(j * col_scale)
    end_col = int(min((j + 1) * col_scale, inverted_edges.shape[1]))

    # Find the maximum value in the corresponding region (0 or 255)
    # Assign -10 only if the maximum is 255 (represents a dark edge)
    max_value = np.max(inverted_edges[start_row:end_row, start_col:end_col])
    #print(max_value)
    small_grid[i, j] = -10 if max_value == 255 else 0

# Print the resulting 100x100 grid with wall and open space markers
print(small_grid.astype(int))

grid = small_grid.astype(int)

with open('grid_rows_3.json', 'w') as f:
    f.write('[\n')
    for row in grid:
        row_str = ', '.join(f'{cell:3}' for cell in row)  # join elements with ', ' and avoid trailing comma
        f.write(f'  [{row_str}],\n')
    f.write(']\n')
