from rob_agi.colored_grid import ColoredGrid

def solve_0f63c0b9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating frames for each non-black color.
    
    The transformation follows these rules:
    1. Colors are processed from top to bottom based on their first appearance.
    2. Each color creates a frame-like structure:
       - For a single color, it fills the entire grid.
       - For multiple colors:
         - The topmost color fills at least the top row and extends down.
         - Middle colors (if any) fill at least their top row and extend down.
         - The bottommost color fills at least the bottom row and extends up.
    3. Vertical lines for each color extend from its start to its end boundary.
    4. The frame structure adapts to the spacing between colors in the input.
    5. The interior of each frame remains black.
    
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
    
    # Process each color
    for i, (color, start_row) in enumerate(colors):
        is_first = i == 0
        is_last = i == len(colors) - 1
        
        # Determine boundaries
        top_boundary = start_row if not is_first else 0
        bottom_boundary = 14 if is_last else (colors[i+1][1] - 1 if i+1 < len(colors) else 13)
        
        # Fill horizontal rows
        output_grid[top_boundary] = [color] * 15
        if is_first:
            output_grid[top_boundary + 1] = [color] * 15
        if is_last:
            output_grid[14] = [color] * 15
            output_grid[13] = [color] * 15
        
        # Fill vertical lines and extend color
        for row in range(top_boundary, bottom_boundary + 1):
            output_grid[row][0] = color
            output_grid[row][14] = color
            if row == top_boundary or row == bottom_boundary:
                output_grid[row] = [color] * 15
        
        # Extend color to fill large gaps
        if not is_last and i + 1 < len(colors):
            next_start = colors[i+1][1]
            gap = next_start - bottom_boundary - 1
            if gap > 1:
                extend_to = bottom_boundary + gap // 2
                for row in range(bottom_boundary + 1, extend_to):
                    output_grid[row][0] = color
                    output_grid[row][14] = color
    
    return ColoredGrid(values=output_grid)
