from rob_agi.colored_grid import ColoredGrid

def solve_0c9aba6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 13x4 input grid into a 6x4 output grid based on the following rules:
    1. Only considers the top 6 rows of the input grid.
    2. For each cell in the output grid:
       - Count the number of red (2) squares in a 2x2 block starting from the corresponding position in the input grid.
       - If the count is odd, set the output cell to sky blue (8).
       - If the count is even (including 0), set the output cell to black (0).
    3. Handle edge cases where a full 2x2 block isn't available (last row or column).
    4. Returns the resulting 6x4 grid.
    """
    def count_red_in_block(r: int, c: int) -> int:
        count = 0
        for i in range(2):
            for j in range(2):
                if r + i < 6 and c + j < 4:  # Check if within bounds
                    if input_grid.values[r + i][c + j] == 2:  # Check if red
                        count += 1
        return count

    # Create a new 6x4 ColoredGrid for the output, initially filled with black (0)
    output_grid = ColoredGrid(values=[[0 for _ in range(4)] for _ in range(6)])

    # Process each cell in the output grid
    for r in range(6):
        for c in range(4):
            red_count = count_red_in_block(r, c)
            if red_count % 2 == 1:  # odd count
                output_grid.values[r][c] = 8  # sky blue
            # If even, leave as 0 (black)

    return output_grid
