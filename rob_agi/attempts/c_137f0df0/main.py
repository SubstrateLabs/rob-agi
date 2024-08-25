from rob_agi.colored_grid import ColoredGrid

def solve_137f0df0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a red background, preserving gray blocks,
    and adding blue squares at regular intervals on the edges.
    
    1. Creates a full red (2) background.
    2. Transfers the original gray (5) blocks to their corresponding positions.
    3. Adds blue (1) squares at regular intervals on the grid edges based on the gray block pattern.
    4. Fills remaining edge spaces with black (0).
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified pattern.
    """
    def find_gray_pattern(grid):
        rows, cols = len(grid), len(grid[0])
        gray_rows, gray_cols = set(), set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 5:
                    gray_rows.add(r)
                    gray_cols.add(c)
        return sorted(gray_rows), sorted(gray_cols)

    def create_transformed_grid(input_grid, gray_rows, gray_cols):
        rows, cols = len(input_grid), len(input_grid[0])
        new_grid = [[2 for _ in range(cols)] for _ in range(rows)]  # Fill with red

        # Transfer gray blocks
        for r in range(rows):
            for c in range(cols):
                if input_grid[r][c] == 5:
                    new_grid[r][c] = 5

        # Add blue squares on edges
        for r in [0, rows-1]:
            for c in gray_cols:
                new_grid[r][c] = 1
        for c in [0, cols-1]:
            for r in gray_rows:
                new_grid[r][c] = 1

        # Fill remaining edge cells with black
        for i in range(cols):
            if new_grid[0][i] == 2:
                new_grid[0][i] = 0
            if new_grid[rows-1][i] == 2:
                new_grid[rows-1][i] = 0
        for i in range(rows):
            if new_grid[i][0] == 2:
                new_grid[i][0] = 0
            if new_grid[i][cols-1] == 2:
                new_grid[i][cols-1] = 0

        return new_grid

    input_values = input_grid.values
    gray_rows, gray_cols = find_gray_pattern(input_values)
    new_grid = create_transformed_grid(input_values, gray_rows, gray_cols)
    
    return ColoredGrid(values=new_grid)
