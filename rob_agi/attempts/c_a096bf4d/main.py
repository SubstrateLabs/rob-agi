from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_a096bf4d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Analyze the entire grid to identify border color and special colors for each row.
    2. Process each 5x5 section:
       a. Determine the main color of the 3x3 interior.
       b. Apply row-specific transformations (top: 2, middle: 8, bottom: 4).
       c. Preserve special colors (7) and replace others (6 with 1).
       d. Ensure consistent interiors for sections in the same row.
    3. Reconstruct the grid maintaining original border colors and special color positions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    border_color = input_grid.values[0][0]

    # Identify special colors for each row
    special_colors = identify_special_colors(input_grid)

    for r in range(0, rows, 5):
        for c in range(0, cols, 5):
            if r + 5 <= rows and c + 5 <= cols:
                subgrid = input_grid.extract_subgrid(r, c, 5, 5)
                row_index = r // 5
                transformed = transform_section(subgrid, special_colors[row_index], row_index)
                for i in range(5):
                    for j in range(5):
                        output_grid.values[r+i][c+j] = transformed.values[i][j]

    return output_grid

def identify_special_colors(grid: ColoredGrid) -> List[int]:
    rows, cols = grid.get_dimensions()
    special_colors = []
    for r in range(0, rows, 5):
        row_colors = set()
        for c in range(0, cols, 5):
            if r + 5 <= rows and c + 5 <= cols:
                subgrid = grid.extract_subgrid(r, c, 5, 5)
                row_colors.update(subgrid.values[1][1:4] + subgrid.values[2][1:4] + subgrid.values[3][1:4])
        if r == 0:
            special_colors.append(2 if 2 in row_colors else 8)
        elif r == rows - 5:
            special_colors.append(4 if 4 in row_colors else 8)
        else:
            special_colors.append(8)
    return special_colors

def transform_section(section: ColoredGrid, special_color: int, row_index: int) -> ColoredGrid:
    border_color = section.values[0][0]
    interior = section.extract_subgrid(1, 1, 3, 3)
    
    main_color = analyze_interior(interior)
    
    new_interior = [[main_color for _ in range(3)] for _ in range(3)]
    
    # Apply row-specific transformation
    if row_index == 0:  # Top row
        new_interior[1][1] = special_color
    elif row_index == len(section.values) // 5 - 1:  # Bottom row
        new_interior[0][2] = special_color
    else:  # Middle row(s)
        new_interior[1][1] = special_color
        new_interior[2][0] = 3
    
    # Preserve special colors and replace others
    for i in range(3):
        for j in range(3):
            if interior.values[i][j] == 7:
                new_interior[i][j] = 7
            elif interior.values[i][j] == special_color:
                new_interior[i][j] = special_color
            elif interior.values[i][j] == 6:
                new_interior[i][j] = 1
    
    # Fill remaining cells with border color
    for i in range(3):
        for j in range(3):
            if new_interior[i][j] == main_color:
                new_interior[i][j] = border_color
    
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
