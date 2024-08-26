from rob_agi.colored_grid import ColoredGrid

def solve_54db823b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by removing all colored squares to the left of (and including) 
    the rightmost column that is either all black or has an all-black column to its right.
    
    1. Finds the dividing line (rightmost column satisfying the condition).
    2. Creates a new grid with the same dimensions as the input.
    3. Copies the input grid, setting all cells to the left of the dividing line to black (0).
    4. Returns the new grid.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Find the dividing line
    dividing_line = -1
    for col in range(cols - 1):
        if all(input_grid.values[row][col] == 0 for row in range(rows)) or \
           all(input_grid.values[row][col + 1] == 0 for row in range(rows)):
            dividing_line = col

    # Step 2: Create a new ColoredGrid
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Step 3: Copy and modify the grid
    for row in range(rows):
        for col in range(cols):
            if col <= dividing_line:
                new_grid.values[row][col] = 0
            else:
                new_grid.values[row][col] = input_grid.values[row][col]

    # Step 4: Return the new grid
    return new_grid
