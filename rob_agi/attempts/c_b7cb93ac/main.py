from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_b7cb93ac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 3x4 output grid based on color patterns.
    
    1. Analyzes the input grid to determine the three most common colors (including sky blue).
    2. Creates a 3x4 grid with the following layout:
       - Most common color forms either a 3x2 rectangle on the right or a cross shape.
       - Second most common color fills the remaining spaces in the top and bottom rows.
       - Third most common color (or sky blue if present) goes in the corners.
    3. Ensures vertical symmetry between the top and bottom rows.
    4. The middle row is always filled with the most common color.
    
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
    
    # Step 3: Determine layout
    first_color, second_color, third_color = [color for color, _ in top_colors]
    if color_counts[second_color] > color_counts[third_color]:
        output = create_3x2_rectangle_layout(first_color, second_color, third_color)
    else:
        output = create_cross_shape_layout(first_color, second_color, third_color)
    
    # Step 4: Ensure middle row is filled with the most common color
    output[1] = [first_color] * 4
    
    # Step 5: Create and return ColoredGrid
    return ColoredGrid(values=output)

def create_3x2_rectangle_layout(first: int, second: int, third: int) -> List[List[int]]:
    return [
        [third, first, first, first],
        [second, first, first, first],
        [third, first, first, first]
    ]

def create_cross_shape_layout(first: int, second: int, third: int) -> List[List[int]]:
    return [
        [third, first, second, second],
        [first, first, first, first],
        [third, first, second, second]
    ]
