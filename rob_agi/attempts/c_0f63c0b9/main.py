from rob_agi.colored_grid import ColoredGrid

def solve_0f63c0b9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating frames for each non-black color.
    
    The transformation follows these rules:
    1. Colors are processed from top to bottom based on their first appearance.
    2. Each color creates a frame-like structure:
       - The topmost color fills the top two rows completely and extends down with vertical lines.
       - Middle colors (if any) have one full row in the middle and vertical lines extending to their boundaries.
       - The bottommost color fills the bottom two rows completely and extends up with vertical lines.
    3. There's always at least one black row between color sections.
    4. Each color section has a minimum height of 3 rows.
    5. Space is distributed evenly among colors, with preference given to top and bottom colors.
    
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
    
    # Handle single color case
    if len(colors) == 1:
        return ColoredGrid(values=[[colors[0][0] for _ in range(15)] for _ in range(15)])
    
    # Calculate space distribution
    num_colors = len(colors)
    min_space_per_color = 3
    total_space = 15 - (num_colors - 1)  # Reserve space for black separators
    base_height = max(min_space_per_color, total_space // num_colors)
    extra_space = total_space % num_colors
    
    # Distribute space
    color_spaces = [base_height + (1 if i < extra_space else 0) for i in range(num_colors)]
    
    # Process each color
    current_row = 0
    for i, (color, _) in enumerate(colors):
        is_first = i == 0
        is_last = i == num_colors - 1
        
        # Determine boundaries
        top_boundary = current_row
        bottom_boundary = current_row + color_spaces[i] - 1
        
        # Fill horizontal rows
        if is_first:
            output_grid[top_boundary] = [color] * 15
            output_grid[top_boundary + 1] = [color] * 15
        elif is_last:
            output_grid[bottom_boundary - 1] = [color] * 15
            output_grid[bottom_boundary] = [color] * 15
        else:
            mid_row = (top_boundary + bottom_boundary) // 2
            output_grid[mid_row] = [color] * 15
        
        # Fill vertical lines
        for row in range(top_boundary, bottom_boundary + 1):
            output_grid[row][0] = color
            output_grid[row][14] = color
        
        current_row = bottom_boundary + 2  # +2 to include the black separator row
    
    return ColoredGrid(values=output_grid)
