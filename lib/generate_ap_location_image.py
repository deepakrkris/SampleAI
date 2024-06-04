import numpy as np
import json
from PIL import Image
# Import colormap function from matplotlib.cm
from matplotlib.cm import get_cmap
import io

def grid_to_image(grid, colormap="viridis", scale_factor=2, icon_position=(0, 0)):
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

  icon_path="lib/google_loc_icon_2.png"

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

  if icon_path is not None:
    try:
      # Load the icon image
      icon_image = Image.open(icon_path)

      # Get the icon size
      icon_width, icon_height = icon_image.size

      # Calculate the top-left corner coordinates for the icon placement
      grid_width, grid_height = image.size
      x_pos, y_pos = icon_position
      x_image = x_pos * (grid_width // 100)
      y_image = y_pos * (grid_height // 100)

      adjusted_x = max(0, min(x_image, grid_width - icon_width))  # Clamp x position
      adjusted_y = max(0, min(y_image, grid_height - icon_height))  # Clamp y position

      # Paste the icon onto the grid image
      image.paste(icon_image, (adjusted_x, adjusted_y))
    except FileNotFoundError:
      print(f"Error: Icon file not found at {icon_path}")

  img_byte_arr = io.BytesIO()
  image.save(img_byte_arr, format='PNG')
  img_byte_arr.seek(0)

  return img_byte_arr