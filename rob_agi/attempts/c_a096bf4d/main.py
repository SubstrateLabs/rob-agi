from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_a096bf4d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Identify the border color for each 5x5 section.
    2. Find the highest-numbered non-border color in the entire grid.
    3. For each 5x5 section:
       a. Determine the main color (most frequent) in the 3x3 interior.
       b. Create a new 3x3 interior with:
          - Top-left 2x2 filled with the main color
          - Bottom-right of the 2x2 filled with the highest-numbered non-border color
          - Remaining cells filled with the border color
       c. If the original interior already contains the highest non-border color, keep its position.
    4. Reconstruct the grid maintaining the original border colors.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find the highest non-border color in the entire grid
    border_color = input_grid.values[0][0]
    all_colors = set(cell for row in input_grid.values for cell in row if cell != border_color)
    highest_color = max(all_colors) if all_colors else border_color

    for r in range(0, rows, 5):
        for c in range(0, cols, 5):
            if r + 5 <= rows and c + 5 <= cols:
                subgrid = input_grid.extract_subgrid(r, c, 5, 5)
                transformed = transform_section(subgrid, highest_color)
                for i in range(5):
                    for j in range(5):
                        output_grid.values[r+i][c+j] = transformed.values[i][j]

    return output_grid

def transform_section(section: ColoredGrid, highest_color: int) -> ColoredGrid:
    border_color = section.values[0][0]
    interior = section.extract_subgrid(1, 1, 3, 3)
    
    main_color = analyze_interior(interior)
    
    new_interior = [
        [main_color, main_color, border_color],
        [main_color, highest_color, border_color],
        [border_color, border_color, border_color]
    ]
    
    # Check if the highest color already exists in the original interior
    for i in range(3):
        for j in range(3):
            if interior.values[i][j] == highest_color:
                new_interior[i][j] = highest_color
    
    new_section = section.deep_copy()
    for i in range(3):
        for j in range(3):
            new_section.values[i+1][j+1] = new_interior[i][j]
    
    return new_section

def analyze_interior(interior: ColoredGrid) -> int:
    colors = [cell for row in interior.values for cell in row]
    color_counts = Counter(colors)
    main_color = max(color_counts, key=color_counts.get)
    return main_color
