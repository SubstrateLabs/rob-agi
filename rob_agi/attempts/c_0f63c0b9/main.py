from rob_agi.colored_grid import ColoredGrid

def solve_0f63c0b9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating frames for each non-black color.
    
    The transformation follows these rules:
    1. Colors are processed from top to bottom based on their first appearance.
    2. Each color creates a frame-like structure:
       - The top row of its section is filled with the color.
       - The leftmost and rightmost columns are filled from its start to the next color's start.
    3. The topmost color also fills the row below its top row.
    4. The bottommost color fills the entire bottom row of the grid.
    
    Args:
    input_grid (ColoredGrid): The input 15x15 grid with scattered colored squares.
    
    Returns:
    ColoredGrid: The transformed 15x15 grid with color frames.
    """
    output_grid = [[0 for _ in range(15)] for _ in range(15)]
    
    # Scan and sort colors
    colors = []
    for row in range(15):
        for col in range(15):
            if input_grid.values[row][col] != 0:
                colors.append((input_grid.values[row][col], row))
    colors.sort(key=lambda x: x[1])  # Sort by row
    
    # Determine section boundaries
    sections = []
    for i, (color, start_row) in enumerate(colors):
        end_row = colors[i+1][1] - 1 if i < len(colors) - 1 else 14
        sections.append((color, start_row, end_row))
    
    # Process each color
    for i, (color, start_row, end_row) in enumerate(sections):
        # Fill top row
        output_grid[start_row] = [color] * 15
        
        # Fill vertical lines
        for row in range(start_row, end_row + 1):
            output_grid[row][0] = color
            output_grid[row][14] = color
        
        if i == 0:  # Topmost color
            output_grid[start_row + 1] = [color] * 15
        
        if i == len(sections) - 1:  # Bottommost color
            output_grid[14] = [color] * 15
    
    return ColoredGrid(values=output_grid)
