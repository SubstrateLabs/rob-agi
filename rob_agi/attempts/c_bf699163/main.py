from rob_agi.colored_grid import ColoredGrid

def solve_bf699163(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the bf699163 challenge by finding the first valid 3x3 pattern in the input grid
    and returning a new 3x3 grid based on that pattern.

    A valid pattern is a 3x3 subgrid with a gray (5) center and all surrounding cells
    of the same non-gray color. The function returns a new 3x3 grid with the same pattern
    as the first valid one found in the input grid, scanning from top-left to bottom-right.
    The function checks all possible 3x3 subgrids, including those at the edges of the input grid.

    Args:
    input_grid (ColoredGrid): The input grid to analyze.

    Returns:
    ColoredGrid: A 3x3 grid representing the first valid pattern found, or None if no valid pattern is found.
    """
    def is_valid_pattern(grid, row, col):
        center_color = grid.values[row][col]
        if center_color != 5:  # Center must be gray
            return None
        
        surrounding_color = None
        for i in range(max(0, row-1), min(rows, row+2)):
            for j in range(max(0, col-1), min(cols, col+2)):
                if i == row and j == col:
                    continue
                current_color = grid.values[i][j]
                if current_color == 5:  # Surrounding cells can't be gray
                    return None
                if surrounding_color is None:
                    surrounding_color = current_color
                elif current_color != surrounding_color:
                    return None
        
        return surrounding_color

    rows, cols = input_grid.get_dimensions()

    for row in range(rows):
        for col in range(cols):
            color = is_valid_pattern(input_grid, row, col)
            if color is not None:
                # Return the first valid pattern found
                return ColoredGrid(values=[
                    [color, color, color],
                    [color, 5, color],
                    [color, color, color]
                ])

    return None  # No valid pattern found
