from rob_agi.colored_grid import ColoredGrid

def solve_95990924(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 95990924 challenge by identifying 2x2 squares of 5s and placing numbers 1-4
    in diagonally adjacent cells around each square.

    The function creates a deep copy of the input grid, identifies all 2x2 squares of 5s,
    and places numbers 1-4 in the diagonally adjacent cells for each square. It preserves
    all original numbers in the grid and only places new numbers in empty (0) cells.

    Args:
        input_grid (ColoredGrid): The input grid to be processed.

    Returns:
        ColoredGrid: The processed grid with numbers 1-4 placed around 2x2 squares of 5s.
    """
    def safe_place(grid: ColoredGrid, row: int, col: int, value: int) -> None:
        """Safely place a value in the grid if the cell is empty and within bounds."""
        height, width = grid.get_dimensions()
        if 0 <= row < height and 0 <= col < width and grid.get_cell(row, col) == 0:
            grid.set_cell(row, col, value)

    def process_square(grid: ColoredGrid, row: int, col: int) -> None:
        """Place numbers 1-4 in diagonal positions around a 2x2 square of 5s."""
        safe_place(grid, row-1, col-1, 1)
        safe_place(grid, row-1, col+2, 2)
        safe_place(grid, row+2, col-1, 3)
        safe_place(grid, row+2, col+2, 4)

    # Create a deep copy of the input grid
    result = input_grid.deep_copy()
    height, width = result.get_dimensions()

    # Identify and process all 2x2 squares of 5s
    for row in range(height - 1):
        for col in range(width - 1):
            if all(result.get_cell(row+i, col+j) == 5 for i in range(2) for j in range(2)):
                process_square(result, row, col)

    return result
