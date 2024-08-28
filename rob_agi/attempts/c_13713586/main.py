from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_13713586(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions column-wise, first vertically and then horizontally.
    The expansion respects original boundaries and the order of colors from left to right.

    The algorithm works as follows:
    1. Create a copy of the input grid.
    2. Identify all colored positions (excluding black and gray) and sort them by column, then row.
    3. Process each column from left to right:
       a. Expand colors vertically within the column.
       b. Expand colors horizontally to the right, limited by vertical expansions and previous column expansions.
    4. Preserve all original non-black colors and gray borders.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def expand_vertically(grid: ColoredGrid, row: int, col: int, color: int) -> Tuple[int, int]:
        top, bottom = row, row
        # Expand upwards
        for r in range(row-1, -1, -1):
            if input_grid.values[r][col] != 0:
                break
            grid.values[r][col] = color
            top = r
        # Expand downwards
        for r in range(row+1, rows):
            if input_grid.values[r][col] != 0:
                break
            grid.values[r][col] = color
            bottom = r
        return top, bottom

    def expand_horizontally(grid: ColoredGrid, row: int, col: int, color: int, right_limit: int):
        for c in range(col+1, right_limit):
            if input_grid.values[row][c] != 0 or grid.values[row][c] not in [0, color]:
                break
            grid.values[row][c] = color

    colored_positions = [(r, c, grid.values[r][c]) for c in range(cols) for r in range(rows) 
                         if grid.values[r][c] not in [0, 5]]
    colored_positions.sort(key=lambda x: (x[1], x[0]))  # Sort by column, then row

    # Process columns from left to right
    for col in range(cols):
        col_colors = [pos for pos in colored_positions if pos[1] == col]
        vertical_expansions: Dict[int, Tuple[int, int]] = {}
        
        # Vertical expansion
        for row, _, color in col_colors:
            top, bottom = expand_vertically(grid, row, col, color)
            vertical_expansions[color] = (top, bottom)
        
        # Horizontal expansion
        for row in range(rows):
            for _, _, color in col_colors:
                if vertical_expansions[color][0] <= row <= vertical_expansions[color][1]:
                    right_limit = cols
                    for next_col in range(col+1, cols):
                        if input_grid.values[row][next_col] != 0:
                            right_limit = next_col
                            break
                    expand_horizontally(grid, row, col, color, right_limit)
                    break  # Move to the next row after expanding the first color found

    # Preserve original non-black colors and gray borders
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                grid.values[r][c] = input_grid.values[r][c]

    return grid
