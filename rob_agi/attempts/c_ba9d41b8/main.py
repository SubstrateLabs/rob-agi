from rob_agi.colored_grid import ColoredGrid

def solve_ba9d41b8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a checkerboard pattern to non-black regions.
    The outer border of each region remains unchanged, while the inner part is filled
    with a checkerboard pattern using the original color and black (0).
    
    The checkerboard pattern is applied based on the global position of each cell,
    ensuring consistent patterning for all regions regardless of their position in the grid.
    Cells where the sum of row and column indices is odd are set to black (0).
    """
    if not input_grid.is_valid:
        raise ValueError("Invalid input grid")

    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def is_border(r: int, c: int, color: int) -> bool:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if output_grid.get_cell(nr, nc) != color:
                    return True
        return False

    for r in range(rows):
        for c in range(cols):
            color = output_grid.get_cell(r, c)
            if color != 0:  # If the cell is not black
                if not is_border(r, c, color):
                    if (r + c) % 2 == 1:  # If sum of row and column is odd
                        output_grid.set_cell(r, c, 0)  # Set to black

    return output_grid
