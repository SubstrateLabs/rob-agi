from rob_agi.colored_grid import ColoredGrid

def solve_ff72ca3e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating red squares around yellow cells.
    
    The function follows these rules:
    1. If there's only one yellow (4) cell:
       - Create the largest possible odd-sized square of red (2) cells centered on the yellow cell.
       - The square should not include any gray (5) cells or go off the grid boundaries.
       - The minimum size of this square is 3x3.
    2. If there are multiple yellow cells:
       - Create a 3x3 square of red (2) cells centered on each yellow cell.
       - Do not overwrite any gray (5) cells or go off the grid boundaries.
    3. Gray (5) cells and the original yellow (4) cells remain unchanged.
    4. All other cells not affected by the above rules remain black (0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    yellow_cells = [(r, c) for r in range(len(output_grid.values)) 
                    for c in range(len(output_grid.values[0])) 
                    if output_grid.values[r][c] == 4]

    if len(yellow_cells) == 1:
        r, c = yellow_cells[0]
        max_size = min(len(output_grid.values), len(output_grid.values[0]))
        max_size = max_size if max_size % 2 == 1 else max_size - 1

        def is_valid_square(size):
            half = size // 2
            for i in range(r - half, r + half + 1):
                for j in range(c - half, c + half + 1):
                    if (i < 0 or i >= len(output_grid.values) or
                        j < 0 or j >= len(output_grid.values[0]) or
                        output_grid.values[i][j] == 5):
                        return False
            return True

        for size in range(max_size, 1, -2):
            if is_valid_square(size):
                half = size // 2
                for i in range(r - half, r + half + 1):
                    for j in range(c - half, c + half + 1):
                        if i != r or j != c:  # Don't overwrite the yellow cell
                            output_grid.values[i][j] = 2
                break
    else:
        for r, c in yellow_cells:
            for i in range(max(0, r - 1), min(len(output_grid.values), r + 2)):
                for j in range(max(0, c - 1), min(len(output_grid.values[0]), c + 2)):
                    if output_grid.values[i][j] != 5 and (i != r or j != c):
                        output_grid.values[i][j] = 2

    return output_grid
