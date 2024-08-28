from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_12997ef3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying unique colors and creating a pattern for each.
    
    The function scans the input grid for colors that are unique in their row and column,
    determines the orientation (horizontal or vertical) based on the alignment of unique colors,
    and creates a new grid where each unique color is represented by a 3x2 pattern.
    
    The 3x2 pattern for each color is as follows:
    [0, color, 0]
    [color, color, 0]
    
    The colors are ordered based on their first appearance in the input grid,
    scanning from top-left to bottom-right.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    def scan_input_grid(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
        unique_colors = []
        for r in range(grid.num_rows):
            for c in range(grid.num_cols):
                color = grid.values[r][c]
                if color != 0:
                    row_unique = all(grid.values[r][i] != color for i in range(grid.num_cols) if i != c)
                    col_unique = all(grid.values[i][c] != color for i in range(grid.num_rows) if i != r)
                    if row_unique and col_unique:
                        unique_colors.append((r, c, color))
        return sorted(unique_colors, key=lambda x: (x[0], x[1]))

    def determine_orientation(unique_colors: List[Tuple[int, int, int]]) -> str:
        rows = set(r for r, _, _ in unique_colors)
        cols = set(c for _, c, _ in unique_colors)
        return 'horizontal' if len(rows) == 1 else 'vertical'

    def create_output_grid(unique_colors: List[Tuple[int, int, int]], orientation: str) -> List[List[int]]:
        if orientation == 'horizontal':
            width, height = len(unique_colors) * 3, 3
        else:
            width, height = 3, len(unique_colors) * 3
    
        output = [[0 for _ in range(width)] for _ in range(height)]
    
        for i, (_, _, color) in enumerate(unique_colors):
            if orientation == 'horizontal':
                start_row, start_col = 0, i * 3
            else:
                start_row, start_col = i * 3, 0
        
            pattern = [
                [0, color, 0],
                [color, color, 0]
            ]
        
            for r in range(2):
                for c in range(3):
                    output[start_row + r][start_col + c] = pattern[r][c]
    
        return output

    unique_colors = scan_input_grid(input_grid)
    orientation = determine_orientation(unique_colors)
    output_values = create_output_grid(unique_colors, orientation)
    return ColoredGrid(values=output_values)
