from rob_agi.colored_grid import ColoredGrid

def solve_2a5f8217(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by updating each non-zero cell's color
    to the maximum color value among its adjacent cells (including diagonals) and itself.

    The transformation happens in a single step:
    1. Analyze the input grid and store the new maximum values.
    2. Create a new grid with the stored maximum values.
    3. Repeat the process once more to handle cascading effects.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with colors updated.
    """
    def transform_once(grid):
        rows, cols = grid.get_dimensions()
        new_values = [[0 for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] != 0:
                    max_color = grid.values[r][c]
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                max_color = max(max_color, grid.values[nr][nc])
                    new_values[r][c] = max_color
                else:
                    new_values[r][c] = 0

        return ColoredGrid(values=new_values)

    # Apply the transformation twice
    intermediate_grid = transform_once(input_grid)
    final_grid = transform_once(intermediate_grid)

    return final_grid
