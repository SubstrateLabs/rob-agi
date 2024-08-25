from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_a096bf4d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules to each 5x5 section:
    1. Identify the main color (most frequent) in the 3x3 interior.
    2. Find the highest-numbered non-main color in the 3x3 interior.
    3. Create a new 3x3 interior with:
       - Top-left 2x2 filled with the main color
       - Bottom-right of the 2x2 filled with the highest-numbered non-main color
       - Remaining cells filled with the border color
    4. Reconstruct the grid maintaining the original border colors.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    for r in range(0, rows, 5):
        for c in range(0, cols, 5):
            if r + 5 <= rows and c + 5 <= cols:
                subgrid = input_grid.extract_subgrid(r, c, 5, 5)
                transformed = transform_section(subgrid)
                for i in range(5):
                    for j in range(5):
                        output_grid.values[r+i][c+j] = transformed.values[i][j]

    return output_grid

def transform_section(section: ColoredGrid) -> ColoredGrid:
    border_color = section.values[0][0]
    interior = section.extract_subgrid(1, 1, 3, 3)
    
    main_color, highest_color = analyze_interior(interior)
    
    new_interior = [
        [main_color, main_color, border_color],
        [main_color, highest_color, border_color],
        [border_color, border_color, border_color]
    ]
    
    new_section = section.deep_copy()
    for i in range(3):
        for j in range(3):
            new_section.values[i+1][j+1] = new_interior[i][j]
    
    return new_section

def analyze_interior(interior: ColoredGrid) -> Tuple[int, int]:
    colors = [cell for row in interior.values for cell in row]
    color_counts = Counter(colors)
    main_color = max(color_counts, key=color_counts.get)
    non_main_colors = [c for c in colors if c != main_color]
    highest_color = max(non_main_colors) if non_main_colors else main_color
    return main_color, highest_color
