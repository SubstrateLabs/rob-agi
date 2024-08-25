from rob_agi.colored_grid import ColoredGrid

def solve_3a301edc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a square border around the main shape.
    
    1. Identifies the main shape in the input grid.
    2. Finds the center color of the shape to use as the border color.
    3. Determines the border thickness based on the largest dimension of the shape.
    4. Creates a new square grid with the original shape centered and adds a border of the center color.
    5. The border thickness is 1 for shapes with largest dimension <= 3, 2 for 3 < largest dimension <= 5,
       and 3 for largest dimension > 5.
    6. Adjusts the result to match the original grid size, preserving black space if necessary.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the bounding box of the main shape
    min_row, min_col, max_row, max_col = rows, cols, 0, 0
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                min_row = min(min_row, r)
                min_col = min(min_col, c)
                max_row = max(max_row, r)
                max_col = max(max_col, c)
    
    # Calculate the largest dimension
    largest_dim = max(max_row - min_row + 1, max_col - min_col + 1)
    
    # Find the center color
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    border_color = input_grid.values[center_row][center_col]
    if border_color == 0:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = center_row + dr, center_col + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] != 0:
                border_color = input_grid.values[nr][nc]
                break
    
    # Determine the border thickness
    if largest_dim <= 3:
        border_thickness = 1
    elif largest_dim <= 5:
        border_thickness = 2
    else:
        border_thickness = 3
    
    # Create the new square grid
    new_size = largest_dim + (2 * border_thickness)
    new_grid = [[border_color for _ in range(new_size)] for _ in range(new_size)]
    
    # Copy the original shape to the center of the new grid
    row_offset = (new_size - (max_row - min_row + 1)) // 2
    col_offset = (new_size - (max_col - min_col + 1)) // 2
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if input_grid.values[r][c] != 0:
                new_grid[r - min_row + row_offset][c - min_col + col_offset] = input_grid.values[r][c]
    
    # Adjust the result to match the original grid size
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    start_row = (rows - new_size) // 2
    start_col = (cols - new_size) // 2
    for r in range(new_size):
        for c in range(new_size):
            if 0 <= start_row + r < rows and 0 <= start_col + c < cols:
                output_grid.values[start_row + r][start_col + c] = new_grid[r][c]
    
    return output_grid
