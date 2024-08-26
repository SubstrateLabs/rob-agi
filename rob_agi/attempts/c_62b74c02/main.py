from rob_agi.colored_grid import ColoredGrid

def solve_62b74c02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending a pattern across the entire width.
    
    1. Identifies the width of the leftmost pattern.
    2. Extracts the pattern without the last column and the last column separately.
    3. Copies the pattern without the last column to the left and right sides of the new grid.
    4. Fills the middle section of each row by repeating the last column of the original pattern.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    # Step 1: Analyze the input grid
    rows, cols = input_grid.get_dimensions()
    pattern_width = 0
    for col in range(cols):
        if all(input_grid.values[row][col] == 0 for row in range(rows)):
            pattern_width = col
            break
    if pattern_width == 0:
        pattern_width = cols

    # Step 2: Extract key parts of the pattern
    pattern_without_last = [row[:pattern_width-1] for row in input_grid.values]
    last_column = [row[pattern_width-1] for row in input_grid.values]

    # Step 3: Create a new grid
    new_values = [[0 for _ in range(cols)] for _ in range(rows)]

    # Step 4: Fill the new grid
    for row in range(rows):
        # Left section
        new_values[row][:pattern_width-1] = pattern_without_last[row]
        
        # Middle section
        middle_width = cols - 2 * (pattern_width - 1)
        new_values[row][pattern_width-1:cols-(pattern_width-1)] = [last_column[row]] * middle_width
        
        # Right section
        new_values[row][cols-(pattern_width-1):] = pattern_without_last[row]

    # Step 5: Create and return a new ColoredGrid
    return ColoredGrid(values=new_values)
