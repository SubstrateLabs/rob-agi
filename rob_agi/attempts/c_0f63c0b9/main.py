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
    3. Vertical lines for each color extend from its start to row 12 or the next color's start.
    4. The bottommost color fills upwards from row 12 to its start row.
    5. There's always at least one row of black between color sections.
    6. The interior of each frame remains black.
    
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
        
        # Fill horizontal rows
        if is_first:
            output_grid[start_row] = [color] * 15
            output_grid[start_row + 1] = [color] * 15
        elif is_last:
            output_grid[13] = [color] * 15
            output_grid[14] = [color] * 15
        else:
            output_grid[start_row] = [color] * 15
        
        # Determine end row for vertical lines
        end_row = 12 if is_last else min(12, colors[i+1][1] - 2)
        
        # Fill vertical lines
        for row in range(start_row + (2 if is_first else 1), end_row + 1):
            output_grid[row][0] = color
            output_grid[row][14] = color
        
        # Special handling for bottommost color
        if is_last:
            for row in range(start_row + 1, 13):
                output_grid[row][0] = color
                output_grid[row][14] = color
    
    return ColoredGrid(values=output_grid)
