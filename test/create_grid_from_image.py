import cv2
import numpy as np

# Load floor plan image and convert to grayscale
images = [cv2.imread("../floor_plans/floor-plan-sample-3.jpg", cv2.IMREAD_GRAYSCALE),
          cv2.imread("../floor_plans/2bhk.jpg", cv2.IMREAD_GRAYSCALE),
          cv2.imread("../floor_plans/sample-floorplan.png", cv2.IMREAD_GRAYSCALE),
          cv2.imread("../floor_plans/fp_grayscale.png", cv2.IMREAD_GRAYSCALE)]

count = 4

for image in images :
  # Apply Canny edge detection (adjust parameters if needed)
  edges = cv2.Canny(image, 50, 150)

  # Invert the edge image (dark edges become white)
  inverted_edges = edges

  # Target grid size
  target_size = (200, 200)

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
      # max_value = np.max(inverted_edges[start_row:end_row, start_col:end_col])
      # Analyze edge strength (higher for thicker lines)
      edge_strength = np.sum(inverted_edges[start_row:end_row, start_col:end_col])

      # Assign values based on classification (adjust thresholds)
      if edge_strength > 3000:  # Adjust threshold for thick lines (walls)
        small_grid[i, j] = -10  # Wall (thick line)
      elif edge_strength > 1000:  # Adjust threshold for thin lines (potential doors)
        small_grid[i, j] = -5  # Door (thin line)
      elif edge_strength > 500:  # Adjust threshold for thin lines (potential doors)
        small_grid[i, j] = -2  # Door (thin line)
      else:
        small_grid[i, j] = 0  # Open space

  # Print the resulting 100x100 grid with wall and open space markers
  print(small_grid.astype(int))

  grid = small_grid.astype(int)

  with open(f'../test/output/grid_rows_{count}.json', 'w') as f:
    f.write('[\n')
    for i, row in enumerate(grid):
        row_str = ', '.join(f'{cell:3}' for cell in row)  # join elements with ', ' and avoid trailing comma
        f.write(f'  [{row_str}]{ "," if i < len(grid) - 1 else "" }\n')
    f.write(']\n')
  count += 1
