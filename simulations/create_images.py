import numpy as np
import json
from PIL import Image
# Import colormap function from matplotlib.cm
from matplotlib.cm import get_cmap

with open('sample_grid.json', 'r') as f:
    grid_from_file = json.load(f)

def grid_to_image(grid, colormap="viridis", scale_factor=3):
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

  return image

def create_images() :
  grid = np.array(grid_from_file)

  # add walls
  for directions in [ (0, 60, 'top'), (50, 100, 'bottom'), (0, 60, 'left'), (50, 100, 'right') ] :
    gap = np.random.randint(15, 25)
    start = np.random.randint(directions[0], directions[0] + gap)
    gap = np.random.randint(15, 25)
    end = np.random.randint(directions[1] - gap, directions[1])
    delta = np.random.randint(35, 65)
    
    if directions[2] == 'top' or directions[2] == 'bottom' :
      grid[start : end , delta] = -10
    elif directions[2] == 'left' or  directions[2] == 'right' :
      grid[delta, start : end] = -10

  # add glasses
  for directions in [ (0, 60, 'vert'), (0, 60, 'hor')] :
    start = np.random.randint(directions[0], directions[1])
    delta = np.random.randint(35, 65)
    length = np.random.randint(35, 55)

    if directions[2] == 'vert' :
      grid[start : start + length , delta] = -1
    elif directions[2] == 'hor' :
      grid[delta, start : start + length] = -1

  # Create the image with a grayscale colormap
  return grid_to_image(grid, colormap="gray")

  # Create the image with a custom colormap
  # color_image = grid_to_image(grid, colormap="plasma", output_file="plasma_grid_image.png")


def create_images_plan2() :
  grid = np.array(grid_from_file)

  for directions in [ (0, 50, 'top'), (40, 100, 'bottom'), (0, 40, 'left'), (40, 100, 'right') ] :
    gap = np.random.randint(15, 25)
    start = np.random.randint(directions[0], directions[0] + gap)
    gap = np.random.randint(15, 25)
    end = np.random.randint(directions[1] - gap, directions[1])
    delta = np.random.randint(25, 75)
    
    if directions[2] == 'top' or directions[2] == 'bottom' :
      grid[start : end , delta] = -10
    elif directions[2] == 'left' or  directions[2] == 'right' :
      grid[delta, start : end] = -10

  # add glasses
  for directions in [ (0, 60, 'vert'), (0, 60, 'hor')] :
    start = np.random.randint(directions[0], directions[1])
    delta = np.random.randint(35, 65)
    length = np.random.randint(35, 55)

    if directions[2] == 'vert' :
      grid[start : start + length // 2 - 5, delta] = -5
      grid[start + length // 2 + 5 : start + length, delta] = -5
      grid[start + length // 2 - 5 : start + length // 2 + 5 , delta] = -2
    elif directions[2] == 'hor' :
      grid[delta, start : start + length // 2 - 5] = -5
      grid[delta, start + length // 2 + 5 : start + length] = -5
      grid[delta, start + length // 2 - 5 : start + length // 2 + 5] = -2

  # Create the image with a grayscale colormap
  return grid_to_image(grid, colormap="gray")

  # Create the image with a custom colormap
  # color_image = grid_to_image(grid, colormap="plasma", output_file="plasma_grid_image.png")


for i in range(130, 150) :
  image = create_images_plan2()
  image.save('images/' + str(i) + '.png')
