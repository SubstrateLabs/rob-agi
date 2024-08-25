from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_0a1d4ef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a large input grid into a smaller 3x3 output grid by identifying
    and arranging the most significant colors.

    The solution follows these steps:
    1. Analyze the input grid to find significant colors based on frequency and region size.
    2. Create a 3x3 color palette from the most significant colors.
    3. Arrange the colors in the output grid to roughly correspond with their positions
       in the input grid while maintaining a balanced distribution.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: A 3x3 grid representing the essence of the input grid.
    """
    # Step 1: Analyze the input grid
    color_counts = Counter(cell for row in input_grid.values for cell in row if cell != 0)
    significant_colors = [color for color, _ in color_counts.most_common(9)]

    # Step 2: Create a 3x3 color palette
    color_palette = significant_colors[:9]
    while len(color_palette) < 9:
        color_palette.append(color_palette[0])  # Repeat colors if needed

    # Step 3: Arrange colors in the output grid
    output_values = arrange_colors(input_grid, color_palette)

    return ColoredGrid(values=output_values)

def arrange_colors(input_grid: ColoredGrid, color_palette: List[int]) -> List[List[int]]:
    """
    Arranges colors in a 3x3 grid based on their positions in the input grid.
    """
    input_rows, input_cols = input_grid.get_dimensions()
    section_height, section_width = input_rows // 3, input_cols // 3

    output_grid = [[0 for _ in range(3)] for _ in range(3)]
    used_colors = set()

    for i in range(3):
        for j in range(3):
            section = input_grid.extract_subgrid(i * section_height, j * section_width, section_height, section_width)
            section_color_counts = Counter(cell for row in section.values for cell in row if cell != 0)
            
            for color in color_palette:
                if color in section_color_counts and color not in used_colors:
                    output_grid[i][j] = color
                    used_colors.add(color)
                    break
            
            if output_grid[i][j] == 0:  # If no color was assigned, use the first unused color
                for color in color_palette:
                    if color not in used_colors:
                        output_grid[i][j] = color
                        used_colors.add(color)
                        break

    return output_grid
