from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0b17323b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 0b17323b challenge by extending a diagonal pattern of blue dots with red dots.
    
    The function finds all blue dots, determines their diagonal pattern, and extends it
    with red dots to create a total of 7 dots (blue + red) along the diagonal. If the
    grid is too small to fit all 7 dots, it adds as many as possible within the grid bounds.
    
    Args:
    input_grid (ColoredGrid): The input grid containing blue dots (color 1)
    
    Returns:
    ColoredGrid: A new grid with the original blue dots and additional red dots (color 2)
    """
    # Step 1: Analyze the input grid
    blue_dots = []
    for region in input_grid.find_connected_regions(1):
        blue_dots.append(region[0])  # Take the first coordinate of each blue region
    
    # Step 2: Determine the diagonal pattern
    blue_dots.sort(key=lambda pos: pos[0] + pos[1])  # Sort from top-left to bottom-right
    if len(blue_dots) > 1:
        spacing = (blue_dots[1][0] - blue_dots[0][0], blue_dots[1][1] - blue_dots[0][1])
    else:
        spacing = (1, 1)  # Default diagonal spacing
    
    # Step 3 & 4: Calculate number of red dots and extend the pattern
    output_grid = input_grid.deep_copy()
    red_dots_to_add = 7 - len(blue_dots)
    last_position = blue_dots[-1] if blue_dots else (0, 0)
    
    for _ in range(red_dots_to_add):
        next_row = last_position[0] + spacing[0]
        next_col = last_position[1] + spacing[1]
        if 0 <= next_row < input_grid.num_rows and 0 <= next_col < input_grid.num_cols:
            output_grid.values[next_row][next_col] = 2  # Add red dot
            last_position = (next_row, next_col)
        else:
            break  # Stop if we go out of bounds
    
    return output_grid
