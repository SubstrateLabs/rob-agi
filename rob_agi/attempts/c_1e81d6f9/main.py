from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_1e81d6f9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving the gray T-shape and limiting other colors to at most 3 occurrences.
    
    1. Preserves the T-shaped gray object.
    2. Processes other colors from top-left to bottom-right, keeping at most 3 occurrences of each color.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Step 1: Preserve the T-shaped gray object
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 5:  # Gray color
                output_grid.values[r][c] = 5
    
    # Step 2: Process other colors
    for color in range(1, 10):
        if color == 5:  # Skip gray
            continue
        count = 0
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == color:
                    if count < 3:
                        output_grid.values[r][c] = color
                        count += 1
                    else:
                        break
            if count == 3:
                break
    
    return output_grid
