from rob_agi.colored_grid import ColoredGrid

def solve_137f0df0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a red grid structure around gray blocks,
    adding blue squares at edge intersections, and maintaining the original gray blocks.
    
    1. Finds the encompassing grid of all gray (5) blocks.
    2. Creates a red (2) grid structure based on the encompassing grid.
    3. Transfers the original gray (5) blocks to their corresponding positions.
    4. Adds blue (1) squares at the intersections of the red grid with the grid edges.
    5. Fills remaining spaces with black (0).
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified pattern.
    """
    def find_encompassing_grid(grid):
        min_row, min_col = len(grid), len(grid[0])
        max_row, max_col = 0, 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 5:
                    min_row = min(min_row, r)
                    min_col = min(min_col, c)
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
        return (min_row, min_col), (max_row, max_col)

    def create_red_grid(grid, top_left, bottom_right):
        new_grid = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        for r in range(top_left[0], bottom_right[0]+2):
            for c in range(top_left[1], bottom_right[1]+2):
                if r == top_left[0] or r == bottom_right[0]+1 or c == top_left[1] or c == bottom_right[1]+1:
                    for i in range(top_left[0], bottom_right[0]+2):
                        new_grid[i][c] = 2
                    for j in range(top_left[1], bottom_right[1]+2):
                        new_grid[r][j] = 2
        return new_grid

    def transfer_gray_blocks(original_grid, new_grid):
        for r in range(len(original_grid)):
            for c in range(len(original_grid[0])):
                if original_grid[r][c] == 5:
                    new_grid[r][c] = 5
        return new_grid

    def add_blue_squares(grid, top_left, bottom_right):
        for r in range(top_left[0], bottom_right[0]+2):
            if r == top_left[0] or r == bottom_right[0]+1:
                grid[r][0] = 1
                grid[r][-1] = 1
        for c in range(top_left[1], bottom_right[1]+2):
            if c == top_left[1] or c == bottom_right[1]+1:
                grid[0][c] = 1
                grid[-1][c] = 1
        return grid

    input_values = input_grid.values
    top_left, bottom_right = find_encompassing_grid(input_values)
    new_grid = create_red_grid(input_values, top_left, bottom_right)
    new_grid = transfer_gray_blocks(input_values, new_grid)
    new_grid = add_blue_squares(new_grid, top_left, bottom_right)
    
    return ColoredGrid(values=new_grid)
