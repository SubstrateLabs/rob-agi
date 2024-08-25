from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_c3202e5a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid by identifying dividing lines, focus color, and creating a new grid
    based on the pattern of the focus color.

    1. Identifies the dividing lines in the input grid.
    2. Determines the section size (3x3 or 4x4).
    3. Finds the focus color (most frequent non-dividing, non-black color).
    4. Analyzes the distribution of the focus color in the grid.
    5. Creates a new grid (5x5 if input sections are 3x3, 3x3 if input sections are 4x4).
    6. Applies a transformation rule to place the focus color in the output grid.

    The transformation aims to capture the essence of the focus color's distribution
    in a simplified geometric pattern, such as an L-shape or diagonal line.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed output grid.
    """
    dividing_color = find_dividing_lines(input_grid)
    section_size = get_section_size(input_grid, dividing_color)
    focus_color = get_focus_color(input_grid, dividing_color)
    
    if section_size == 3:
        output_size = 5
        output_values = expand_pattern(input_grid, dividing_color, focus_color)
    elif section_size == 4:
        output_size = 3
        output_values = contract_pattern(input_grid, dividing_color, focus_color)
    else:
        raise ValueError(f"Unexpected section size: {section_size}")

    return ColoredGrid(values=output_values)

def find_dividing_lines(grid: ColoredGrid) -> int:
    """Identifies the color of the dividing lines."""
    rows, cols = grid.get_dimensions()
    for row in range(rows):
        if all(cell == grid.values[row][0] for cell in grid.values[row]) and grid.values[row][0] != 0:
            return grid.values[row][0]
    raise ValueError("No dividing lines found")

def get_section_size(grid: ColoredGrid, dividing_color: int) -> int:
    """Calculates the size of individual sections."""
    section_size = 0
    for row in grid.values:
        if row[0] == dividing_color:
            return section_size
        section_size += 1
    raise ValueError("Could not determine section size")

def get_focus_color(grid: ColoredGrid, dividing_color: int) -> int:
    """Determines the most frequent non-zero, non-dividing color in the grid."""
    color_counts = {color: 0 for color in range(1, 10) if color != dividing_color}
    for row in grid.values:
        for cell in row:
            if cell != 0 and cell != dividing_color:
                color_counts[cell] = color_counts.get(cell, 0) + 1
    return max(color_counts, key=color_counts.get)

def expand_pattern(grid: ColoredGrid, dividing_color: int, focus_color: int) -> List[List[int]]:
    """Expands the pattern from 3x3 sections to a 5x5 grid."""
    output = [[0 for _ in range(5)] for _ in range(5)]
    sections = [row for row in grid.values if row[0] != dividing_color]
    focus_positions = []
    
    for i, row in enumerate(sections):
        for j, cell in enumerate(row):
            if cell == focus_color:
                focus_positions.append((i % 3, j % 3))
    
    # Apply transformation rule (this is a simple example, adjust as needed)
    for i, j in focus_positions:
        if i == 0 and j == 0:
            output[0][0] = focus_color
        elif i == 0 and j == 2:
            output[0][4] = focus_color
        elif i == 2 and j == 0:
            output[4][0] = focus_color
        elif i == 2 and j == 2:
            output[4][4] = focus_color
        else:
            output[i+1][j+1] = focus_color
    
    return output

def contract_pattern(grid: ColoredGrid, dividing_color: int, focus_color: int) -> List[List[int]]:
    """Contracts the pattern from 4x4 sections to a 3x3 grid."""
    output = [[0 for _ in range(3)] for _ in range(3)]
    sections = [row for row in grid.values if row[0] != dividing_color]
    focus_positions = []
    
    for i, row in enumerate(sections):
        for j, cell in enumerate(row):
            if cell == focus_color:
                focus_positions.append((i % 4, j % 4))
    
    # Count focus color in each quadrant of 4x4 sections
    quadrants = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    for i, j in focus_positions:
        quadrants[(i//2, j//2)] += 1
    
    # Determine the dominant quadrants
    sorted_quadrants = sorted(quadrants.items(), key=lambda x: x[1], reverse=True)
    
    # Apply transformation rule based on dominant quadrants
    if sorted_quadrants[0][1] > sorted_quadrants[1][1]:
        # One dominant quadrant - L-shape
        q = sorted_quadrants[0][0]
        output[q[0]][q[1]] = focus_color
        output[q[0]][2-q[1]] = focus_color
        output[2-q[0]][q[1]] = focus_color
    else:
        # Two or more equally dominant quadrants - diagonal or corners
        for q, _ in sorted_quadrants[:2]:
            output[q[0]*2][q[1]*2] = focus_color
        if sorted_quadrants[0][0][0] != sorted_quadrants[1][0][0] and sorted_quadrants[0][0][1] != sorted_quadrants[1][0][1]:
            output[1][1] = focus_color  # Add center if diagonal
    
    return output
