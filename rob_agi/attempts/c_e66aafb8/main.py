from rob_agi.colored_grid import ColoredGrid
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict

def solve_e66aafb8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e66aafb8 challenge by extracting a representative pattern from the input grid.
    
    The function works as follows:
    1. Preprocess the input grid to create a numpy array and identify non-black cells.
    2. Extract a representative subgrid from the non-black area.
    3. Analyze color frequencies and transitions in the subgrid.
    4. Generate an output grid that captures the essence of the color patterns.
    5. Post-process the output grid to ensure no black cells and appropriate size.
    6. Return the result as a ColoredGrid object.
    
    Returns:
        ColoredGrid: A new grid containing the extracted representative pattern.
    """
    rows, cols = input_grid.get_dimensions()
    grid = np.array([[input_grid.get_cell(r, c) for c in range(cols)] for r in range(rows)])
    non_black_mask = grid != 0
    
    if not np.any(non_black_mask):
        return ColoredGrid(values=[[1, 1], [1, 1]])  # Return a 2x2 blue grid if input is all black
    
    # Extract a representative subgrid
    non_black_rows, non_black_cols = np.where(non_black_mask)
    top, left = non_black_rows.min(), non_black_cols.min()
    bottom, right = non_black_rows.max(), non_black_cols.max()
    subgrid = grid[top:bottom+1, left:right+1]
    
    # Analyze color frequencies and transitions
    color_freq = defaultdict(int)
    transitions = defaultdict(int)
    rows, cols = subgrid.shape
    for r in range(rows):
        for c in range(cols):
            if subgrid[r, c] != 0:
                color_freq[subgrid[r, c]] += 1
                if c < cols - 1 and subgrid[r, c+1] != 0:
                    transitions[(subgrid[r, c], subgrid[r, c+1])] += 1
                if r < rows - 1 and subgrid[r+1, c] != 0:
                    transitions[(subgrid[r, c], subgrid[r+1, c])] += 1
    
    # Determine output size
    total_colors = len(color_freq)
    size = min(max(int(total_colors ** 0.5), 2), 8)  # Between 2x2 and 8x8
    
    # Generate output grid
    output_grid = np.zeros((size, size), dtype=int)
    colors = sorted(color_freq, key=color_freq.get, reverse=True)
    
    # Fill the grid with the most frequent colors first
    color_index = 0
    for r in range(size):
        for c in range(size):
            output_grid[r, c] = colors[color_index]
            color_index = (color_index + 1) % len(colors)
    
    # Apply transitions to make the pattern more representative
    for _ in range(size * 2):
        r, c = np.random.randint(0, size, 2)
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        valid_neighbors = [(nr, nc) for nr, nc in neighbors if 0 <= nr < size and 0 <= nc < size]
        for nr, nc in valid_neighbors:
            if (output_grid[r, c], output_grid[nr, nc]) in transitions:
                output_grid[r, c], output_grid[nr, nc] = output_grid[nr, nc], output_grid[r, c]
                break
    
    return ColoredGrid(values=output_grid.tolist())
