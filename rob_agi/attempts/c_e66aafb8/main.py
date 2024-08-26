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
    3. Analyze color transitions and patterns in the subgrid.
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
    
    # Analyze color transitions
    def analyze_transitions(grid: np.ndarray) -> Dict[Tuple[int, int], int]:
        transitions = defaultdict(int)
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    if c < cols - 1 and grid[r, c+1] != 0:
                        transitions[(grid[r, c], grid[r, c+1])] += 1
                    if r < rows - 1 and grid[r+1, c] != 0:
                        transitions[(grid[r, c], grid[r+1, c])] += 1
        return transitions
    
    transitions = analyze_transitions(subgrid)
    
    # Determine output size
    min_size = 2
    max_size = min(8, min(subgrid.shape) // 2)
    size = min(max(min_size, int(len(transitions) ** 0.5)), max_size)
    
    # Generate output grid
    output_grid = np.zeros((size, size), dtype=int)
    colors = list(set(subgrid.flatten()) - {0})
    color_index = 0
    
    for r in range(size):
        for c in range(size):
            output_grid[r, c] = colors[color_index]
            color_index = (color_index + 1) % len(colors)
    
    # Apply some transitions
    for _ in range(size):
        r, c = np.random.randint(0, size, 2)
        if (output_grid[r, c], output_grid[(r+1)%size, c]) in transitions:
            output_grid[r, c], output_grid[(r+1)%size, c] = output_grid[(r+1)%size, c], output_grid[r, c]
        if (output_grid[r, c], output_grid[r, (c+1)%size]) in transitions:
            output_grid[r, c], output_grid[r, (c+1)%size] = output_grid[r, (c+1)%size], output_grid[r, c]
    
    return ColoredGrid(values=output_grid.tolist())
