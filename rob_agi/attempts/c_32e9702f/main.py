from rob_agi.colored_grid import ColoredGrid

def solve_32e9702f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following rules:
    1. Preserve non-black regions.
    2. Expand non-black regions downwards and to the right with gray (5) cells.
    3. Fill all remaining cells with gray (5).
    4. If there's a 2x2 yellow (4) square in the top-left corner, expand it diagonally by one cell.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[5 for _ in range(cols)] for _ in range(rows)])
    processed = [[False for _ in range(cols)] for _ in range(rows)]

    def flood_fill(r, c, color):
        if (0 <= r < rows and 0 <= c < cols and
            not processed[r][c] and input_grid.values[r][c] == color):
            output_grid.values[r][c] = color
            processed[r][c] = True
            flood_fill(r+1, c, color)
            flood_fill(r, c+1, color)
            flood_fill(r-1, c, color)
            flood_fill(r, c-1, color)

    # Preserve non-black regions
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0 and not processed[r][c]:
                flood_fill(r, c, input_grid.values[r][c])

    # Expand non-black regions
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                if r+1 < rows and not processed[r+1][c]:
                    output_grid.values[r+1][c] = 5
                    processed[r+1][c] = True
                if c+1 < cols and not processed[r][c+1]:
                    output_grid.values[r][c+1] = 5
                    processed[r][c+1] = True

    # Handle yellow expansion
    if (rows > 2 and cols > 2 and
        output_grid.values[0][0] == 4 and output_grid.values[0][1] == 4 and
        output_grid.values[1][0] == 4 and output_grid.values[1][1] == 4):
        output_grid.values[2][2] = 4

    return output_grid
