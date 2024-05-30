import numpy as np
import json
from PIL import Image
# Import colormap function from matplotlib.cm
from matplotlib.cm import get_cmap

with open('grid_rows_2.json', 'r') as f:
    grid_from_file = json.load(f)

def grid_to_image(grid, colormap="viridis", output_file=None, show_image=True, scale_factor=3):
  """
  Converts a grid of numbers to a grayscale or color image.

  Args:
      grid (list or np.ndarray): The grid of numbers to convert.
          - If a list, it should be a list of lists representing the rows and columns.
          - If a NumPy array, it should have a 2D shape (rows, columns).
      colormap (str, optional): The colormap to use for visualization. Defaults to "viridis".
          See matplotlib.cm for available colormaps.
      output_file (str, optional): The filename to save the image to. Defaults to None (no saving).
      show_image (bool, optional): Whether to display the image using PIL.show(). Defaults to True.

  Returns:
      PIL.Image: The created image object.
  """

  # Ensure grid is a NumPy array
  grid = np.array(grid)

  # Rescale values to the range [0, 1] for colormap application
  grid_scaled = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))

  # Create the colormap object
  colormap_obj = get_cmap(colormap)

  # Apply the colormap to the scaled grid, resulting in a 3D array (rows, columns, RGB channels)
  colored_grid = colormap_obj(grid_scaled)

  # Convert the colored array to uint8 for PIL compatibility (0-255 range for each channel)
  colored_grid = (colored_grid * 255).astype(np.uint8)

  # Create the PIL image from the colored array
  image = Image.fromarray(colored_grid)

  if scale_factor > 1:
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    image = image.resize((new_width, new_height), resample=Image.LANCZOS)

  # Optionally save the image
  if output_file is not None:
    image.save(output_file)

  # Optionally display the image
  if show_image:
    image.show()

  return image

# Example usage
grid = grid_from_file  # Sample grid

# Create the image with a grayscale colormap
grayscale_image = grid_to_image(grid, colormap="gray", output_file="monochrome_grid_image.png")

# Create the image with a custom colormap
color_image = grid_to_image(grid, colormap="plasma", output_file="plasma_grid_image.png")
