from rob_agi.colored_grid import ColoredGrid

def solve_f0afb749(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by:
    1. Doubling the grid size
    2. Expanding non-black squares into 2x2 blocks
    3. Adding blue squares (1s) diagonally from top-left to bottom-right
       in all empty positions across all diagonals
    """
    # Step 1: Initialize the output grid
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    output_grid = [[0 for _ in range(output_cols)] for _ in range(output_rows)]

    # Step 2: Expand non-black squares
    for r in range(input_rows):
        for c in range(input_cols):
            if input_grid.values[r][c] != 0:
                color = input_grid.values[r][c]
                output_grid[r*2][c*2] = color
                output_grid[r*2][c*2+1] = color
                output_grid[r*2+1][c*2] = color
                output_grid[r*2+1][c*2+1] = color

    # Step 3: Fill diagonal positions with blue squares
    def fill_diagonals(grid):
        rows, cols = len(grid), len(grid[0])
        
        def is_valid(r, c):
            return 0 <= r < rows and 0 <= c < cols

        # Process diagonals starting from the top edge
        for start_col in range(cols):
            r, c = 0, start_col
            while is_valid(r, c):
                if grid[r][c] == 0:
                    grid[r][c] = 1  # Set to blue
                r += 1
                c += 1

        # Process diagonals starting from the left edge (excluding top-left corner)
        for start_row in range(1, rows):
            r, c = start_row, 0
            while is_valid(r, c):
                if grid[r][c] == 0:
                    grid[r][c] = 1  # Set to blue
                r += 1
                c += 1

        return grid

    output_grid = fill_diagonals(output_grid)

    # Step 4: Create and return the final ColoredGrid
    return ColoredGrid(values=output_grid)
