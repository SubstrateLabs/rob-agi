from rob_agi.colored_grid import ColoredGrid

def solve_0c9aba6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 13x4 input grid into a 6x4 output grid based on the following rules:
    1. For each cell in the output grid:
       - Count the number of red (2) squares in a 2x2 block of the input grid, offset by one row down.
       - If the count is exactly 1, set the output cell to sky blue (8).
       - Otherwise, set the output cell to black (0).
    2. Handle edge cases for the last two rows of the output grid where a full 2x2 block isn't available.
    3. Returns the resulting 6x4 grid.
    """
    def count_red_in_block(r: int, c: int) -> int:
        count = 0
        if r < 4:
            for i in range(2):
                for j in range(2):
                    if input_grid.values[r + i + 1][c + j] == 2:  # Check if red
                        count += 1
        elif r == 4:
            for j in range(2):
                if input_grid.values[5][c + j] == 2:
                    count += 1
            if len(input_grid.values) > 6:
                for j in range(2):
                    if input_grid.values[6][c + j] == 2:
                        count += 1
        elif r == 5:
            for j in range(2):
                if len(input_grid.values) > 6 and input_grid.values[6][c + j] == 2:
                    count += 1
        return count

    # Create a new 6x4 ColoredGrid for the output, initially filled with black (0)
    output_grid = ColoredGrid(values=[[0 for _ in range(4)] for _ in range(6)])

    # Process each cell in the output grid
    for r in range(6):
        for c in range(4):
            red_count = count_red_in_block(r, c)
            if red_count == 1:  # exactly one red square
                output_grid.values[r][c] = 8  # sky blue
            # Otherwise, leave as 0 (black)

    return output_grid
