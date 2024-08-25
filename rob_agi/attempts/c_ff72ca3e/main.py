from rob_agi.colored_grid import ColoredGrid

def solve_ff72ca3e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating red squares around yellow cells.
    
    The function follows these rules:
    1. For each yellow (4) cell:
       - Create the largest possible odd-sized square of red (2) cells centered on the yellow cell.
       - The square should not include any gray (5) cells or go off the grid boundaries.
       - The minimum size of this square is 3x3.
    2. Gray (5) cells and the original yellow (4) cells remain unchanged.
    3. All other cells not affected by the above rules remain black (0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = len(output_grid.values), len(output_grid.values[0])
    yellow_cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 4]

    def is_valid_square(r, c, size):
        half = size // 2
        for i in range(r - half, r + half + 1):
            for j in range(c - half, c + half + 1):
                if (i < 0 or i >= rows or j < 0 or j >= cols or
                    output_grid.values[i][j] == 5 or
                    (output_grid.values[i][j] == 4 and (i != r or j != c))):
                    return False
        return True

    def fill_square(r, c, size):
        half = size // 2
        for i in range(r - half, r + half + 1):
            for j in range(c - half, c + half + 1):
                if output_grid.values[i][j] != 4 and output_grid.values[i][j] != 5:
                    output_grid.values[i][j] = 2

    for r, c in yellow_cells:
        max_size = min(rows, cols)
        max_size = max_size if max_size % 2 == 1 else max_size - 1
        
        for size in range(max_size, 1, -2):
            if is_valid_square(r, c, size):
                fill_square(r, c, size)
                break

    return output_grid
