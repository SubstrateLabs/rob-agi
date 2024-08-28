from rob_agi.colored_grid import ColoredGrid
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict

def solve_e66aafb8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e66aafb8 challenge by extracting a representative pattern from the input grid.
    
    The function works as follows:
    1. Preprocess the input grid to create a numpy array and identify non-black cells.
    2. Analyze global structure and orientation of the input grid.
    3. Extract a representative subgrid from the non-black area.
    4. Analyze color frequencies, transitions, and patterns in the subgrid.
    5. Determine appropriate output grid size and aspect ratio.
    6. Generate an output grid that captures the essence of the color patterns and transitions.
    7. Refine the output grid to match input characteristics.
    8. Return the result as a ColoredGrid object.
    
    Returns:
        ColoredGrid: A new grid containing the extracted representative pattern.
    """
    rows, cols = input_grid.get_dimensions()
    grid = np.array([[input_grid.get_cell(r, c) for c in range(cols)] for r in range(rows)])
    non_black_mask = grid != 0
    
    if not np.any(non_black_mask):
        return ColoredGrid(values=[[1, 1], [1, 1]])  # Return a 2x2 blue grid if input is all black
    
    # Analyze global structure
    non_black_rows, non_black_cols = np.where(non_black_mask)
    top, left = non_black_rows.min(), non_black_cols.min()
    bottom, right = non_black_rows.max(), non_black_cols.max()
    subgrid = grid[top:bottom+1, left:right+1]
    
    # Determine orientation
    height, width = bottom - top + 1, right - left + 1
    is_vertical = height > width
    
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
    
    # Determine output size based on the input subgrid and orientation
    if is_vertical:
        output_rows = min(max(rows // 2, 5), 8)
        output_cols = min(max(cols // 2, 3), 7)
    else:
        output_rows = min(max(rows // 2, 3), 7)
        output_cols = min(max(cols // 2, 5), 8)
    
    # Generate output grid
    output_grid = np.zeros((output_rows, output_cols), dtype=int)
    colors = sorted(color_freq, key=color_freq.get, reverse=True)
    
    # Fill the grid with the most frequent colors first, preserving structure
    for r in range(output_rows):
        for c in range(output_cols):
            input_r = int(r * rows / output_rows)
            input_c = int(c * cols / output_cols)
            if subgrid[input_r, input_c] != 0:
                output_grid[r, c] = subgrid[input_r, input_c]
            else:
                output_grid[r, c] = colors[0]  # Use most frequent color if black
    
    # Apply transitions to make the pattern more representative
    for _ in range(output_rows * output_cols * 2):  # Increased iterations for better pattern representation
        r, c = np.random.randint(0, output_rows), np.random.randint(0, output_cols)
        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        valid_neighbors = [(nr, nc) for nr, nc in neighbors if 0 <= nr < output_rows and 0 <= nc < output_cols]
        for nr, nc in valid_neighbors:
            if (output_grid[r, c], output_grid[nr, nc]) in transitions:
                output_grid[r, c], output_grid[nr, nc] = output_grid[nr, nc], output_grid[r, c]
                break
    
    # Ensure all top colors are represented
    top_colors = colors[:min(5, len(colors))]
    for color in top_colors:
        if color not in output_grid:
            r, c = np.random.randint(0, output_rows), np.random.randint(0, output_cols)
            output_grid[r, c] = color
    
    return ColoredGrid(values=output_grid.tolist())
