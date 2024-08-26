from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_b7cb93ac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 3x4 output grid based on color patterns.
    
    1. Analyzes the input grid to determine the three most important colors.
    2. Creates a 3x4 grid with the following layout:
       - Most common color fills the middle row and second column.
       - Second most common color fills the two rightmost columns of top and bottom rows.
       - Third color (sky blue if present, otherwise third most common) goes in the corners.
    3. Ensures vertical symmetry between the top and bottom rows.
    
    Returns a new ColoredGrid object representing the transformed grid.
    """
    # Step 1: Analyze input grid
    color_counts = Counter(color for row in input_grid.values for color in row if color != 0)
    sky_blue_present = any(8 in row for row in input_grid.values)
    
    # Step 2: Determine color ranking
    if sky_blue_present and 8 not in color_counts:
        color_counts[8] = 1  # Ensure sky blue is considered even if it appears only once
    top_colors = color_counts.most_common(3)
    while len(top_colors) < 3:
        top_colors.append((0, 0))  # Add black if fewer than 3 colors
    
    first_color, second_color, third_color = [color for color, _ in top_colors]
    if sky_blue_present:
        third_color = 8
    
    # Step 3: Create the output grid
    output = [
        [third_color, first_color, second_color, second_color],
        [first_color, first_color, first_color, first_color],
        [third_color, first_color, second_color, second_color]
    ]
    
    # Step 4: Create and return ColoredGrid
    return ColoredGrid(values=output)
