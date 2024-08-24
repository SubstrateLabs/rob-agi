from rob_agi.colored_grid import ColoredGrid

def solve_a5f85a15(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by changing every second non-zero element
    on each diagonal (top-left to bottom-right) to 4 (yellow).
    
    The function processes all diagonals starting from the first column
    and first row, maintaining the original values for odd-numbered
    non-zero elements and changing even-numbered ones to 4.
    """
    def process_diagonal(start_row, start_col):
        row, col = start_row, start_col
        non_zero_count = 0
        while row < height and col < width:
            if grid[row][col] != 0:
                non_zero_count += 1
                if non_zero_count % 2 == 0:
                    grid[row][col] = 4
            row += 1
            col += 1

    height, width = input_grid.get_dimensions()
    grid = [row[:] for row in input_grid.values]  # Deep copy of the input grid

    # Process diagonals starting from the first column
    for start_row in range(height):
        process_diagonal(start_row, 0)

    # Process diagonals starting from the first row (excluding top-left corner)
    for start_col in range(1, width):
        process_diagonal(0, start_col)

    return ColoredGrid(values=grid)
