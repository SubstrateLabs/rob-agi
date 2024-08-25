from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_f4081712(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f4081712 challenge by creating a condensed representation of the input grid.
    
    The solution involves the following steps:
    1. Analyze the input grid for color frequencies and patterns.
    2. Determine the output grid size based on input characteristics.
    3. Create a condensed representation by sampling important areas of the input grid.
    4. Preserve color relationships and key patterns in the output.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A condensed representation of the input grid.
    """
    # Step 1: Analyze the input grid
    color_freq = count_colors(input_grid)
    transitions = input_grid.find_color_transitions()
    
    # Step 2: Determine output grid size
    output_size = determine_output_size(input_grid, color_freq)
    
    # Step 3 & 4: Create condensed representation
    output_values = create_condensed_grid(input_grid, output_size, color_freq, transitions)
    
    return ColoredGrid(values=output_values)

def count_colors(grid: ColoredGrid) -> Counter:
    return Counter(cell for row in grid.values for cell in row)

def determine_output_size(grid: ColoredGrid, color_freq: Counter) -> Tuple[int, int]:
    unique_colors = len(color_freq)
    input_size = len(grid.values)
    
    # Simple heuristic: output size is between 3x3 and 8x8, scaled by unique colors
    size = max(3, min(8, unique_colors + 2))
    return (size, size)

def create_condensed_grid(grid: ColoredGrid, output_size: Tuple[int, int], color_freq: Counter, transitions: List[Tuple[int, int, int, int]]) -> List[List[int]]:
    input_rows, input_cols = len(grid.values), len(grid.values[0])
    output_rows, output_cols = output_size
    
    # Create an empty output grid
    output = [[0 for _ in range(output_cols)] for _ in range(output_rows)]
    
    # Sample from input grid, prioritizing areas with transitions
    for i in range(output_rows):
        for j in range(output_cols):
            input_row = int(i * input_rows / output_rows)
            input_col = int(j * input_cols / output_cols)
            
            # Check if there's a transition nearby
            for tr, tc, _, _ in transitions:
                if abs(tr - input_row) <= 1 and abs(tc - input_col) <= 1:
                    output[i][j] = grid.values[tr][tc]
                    break
            else:
                # If no transition nearby, just sample from the input grid
                output[i][j] = grid.values[input_row][input_col]
    
    return output
