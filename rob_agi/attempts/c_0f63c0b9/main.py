from rob_agi.colored_grid import ColoredGrid

def solve_0f63c0b9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating frames for each non-black color.
    
    The transformation follows these rules:
    1. Colors are processed from top to bottom based on their first appearance.
    2. Each color creates a frame-like structure:
       - For the topmost color, the top two rows are filled.
       - For middle colors, only the top row of its section is filled.
       - For the bottommost color, the bottom two rows are filled.
       - The leftmost and rightmost columns are filled from its start to the next color's start or row 12.
    3. The interior of each frame remains black.
    
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
            if input_grid.values[row][col] != 0 and input_grid.values[row][col] not in [c[0] for c in colors]:
                colors.append((input_grid.values[row][col], row))
    colors.sort(key=lambda x: x[1])  # Sort by row
    
    # Process each color
    for i, (color, start_row) in enumerate(colors):
        is_first = i == 0
        is_last = i == len(colors) - 1
        
        # Determine end row for vertical lines
        end_row = 12 if is_last else colors[i+1][1] - 1
        
        # Fill top row(s)
        output_grid[start_row] = [color] * 15
        if is_first:
            output_grid[start_row + 1] = [color] * 15
        
        # Fill vertical lines
        for row in range(start_row + (2 if is_first else 1), end_row + 1):
            output_grid[row][0] = color
            output_grid[row][14] = color
        
        # Fill bottom rows for last color
        if is_last:
            output_grid[13] = [color] * 15
            output_grid[14] = [color] * 15
    
    return ColoredGrid(values=output_grid)
