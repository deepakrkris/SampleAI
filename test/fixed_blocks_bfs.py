import numpy as np
import json
from collections import deque

def count_walls_between_blocks(grid, block_size, block_x1, block_y1, block_x2, block_y2):
    walls = 0
    if block_x1 == block_x2:
        # Moving horizontally
        for i in range(block_size):
            if grid[block_x1 * block_size + i, block_y1 * block_size - 1] == -10 or grid[block_x1 * block_size + i, block_y1 * block_size] == -10:
                walls += 1
    elif block_y1 == block_y2:
        # Moving vertically
        for i in range(block_size):
            if grid[block_x1 * block_size - 1, block_y1 * block_size + i] == -10 or grid[block_x1 * block_size, block_y1 * block_size + i] == -10:
                walls += 1
    return walls



def block_bfs_propagate(grid, block_size, start_block_x, start_block_y):
    rows, cols = grid.shape
    block_rows, block_cols = rows // block_size, cols // block_size
    strength_grid = np.zeros((block_rows, block_cols))
    visited = set()
    queue = deque([(start_block_x, start_block_y, 10)])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

    while queue:
        x, y, strength = queue.popleft()
        
        visited.add((x, y))

        if strength <= 0:
            continue

        if strength_grid[x, y] < strength:
            strength_grid[x, y] = strength
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < block_rows and 0 <= ny < block_cols and (nx, ny) not in visited:
                walls = count_walls_between_blocks(grid, block_size, x, y, nx, ny)
                reduced_strength = strength - walls - 1  # 1 for distance and walls for walls
                if (nx, ny) not in visited:
                    queue.append((nx, ny, reduced_strength))
    
    return strength_grid

def create_blocks(grid, block_size):
    rows, cols = grid.shape
    blocks = []
    for start_x in range(0, rows, block_size):
        for start_y in range(0, cols, block_size):
            blocks.append((start_x, start_y))
    return blocks

def find_best_block(grid, block_size):
    blocks = create_blocks(grid, block_size)
    best_block = None
    best_coverage = -np.inf
    
    for block_start_x, block_start_y in blocks:
        block_x = block_start_x // block_size
        block_y = block_start_y // block_size
        strength_grid = block_bfs_propagate(grid, block_size, block_x, block_y)
        
        block_coverage = (np.count_nonzero(strength_grid > 0) / strength_grid.size) * 100
        
        if block_coverage > best_coverage:
            best_coverage = block_coverage
            best_block = (block_start_x, block_start_y)
    
    return best_block, best_coverage


grid_files = ['grid_rows_2.json', 'grid_rows_4.json', 'grid_rows_5.json', 'grid_rows_6.json', 'grid_rows_7.json']

for file in grid_files :
    with open('../test/output/' + file, 'r') as f:
      grid_from_file = json.load(f)

      # Example usage
      grid = np.array(grid_from_file)

      block_size = 10
      best_block, best_coverage = find_best_block(grid, block_size)

      print(f"Best Block: {best_block} with Coverage: {best_coverage}")
