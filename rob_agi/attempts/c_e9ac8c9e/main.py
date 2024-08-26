from rob_agi.colored_grid import ColoredGrid

def solve_e9ac8c9e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by identifying a gray square and four colored squares,
    then creating a new compact arrangement of those colors.
    
    1. Find the gray (5) square in the input.
    2. Identify the four colored squares around the gray square.
    3. Create a new grid with the colored squares arranged in a 2x2 formation,
       each color occupying a quarter of the space previously taken by the gray square.
    4. Center this new formation in the output grid.
    
    The gray square is removed in the output, and the colored squares are expanded
    to fill the space in a compact, symmetrical arrangement.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the gray square
    gray_top, gray_left, gray_size = find_gray_square(input_grid)
    
    # Find the four colors
    colors = find_colors(input_grid, gray_top, gray_left, gray_size)
    
    # Create the output grid
    output = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Calculate the new block size and starting position
    block_size = gray_size // 2
    start_row = (rows - gray_size) // 2
    start_col = (cols - gray_size) // 2
    
    # Fill the new formation
    for i in range(2):
        for j in range(2):
            color = colors[i * 2 + j]
            for r in range(block_size):
                for c in range(block_size):
                    output.values[start_row + i * block_size + r][start_col + j * block_size + c] = color
    
    return output

def find_gray_square(grid: ColoredGrid) -> tuple:
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 5:
                size = 1
                while r + size < grid.num_rows and c + size < grid.num_cols and grid.values[r + size][c + size] == 5:
                    size += 1
                return r, c, size
    return 0, 0, 0  # Default if no gray square found

def find_colors(grid: ColoredGrid, top: int, left: int, size: int) -> list:
    colors = [0, 0, 0, 0]  # TL, TR, BL, BR
    mid = size // 2
    
    # Check corners first, then midpoints if corner is black
    positions = [
        (top - 1, left - 1, 0),  # Top-left
        (top - 1, left + size, 1),  # Top-right
        (top + size, left - 1, 2),  # Bottom-left
        (top + size, left + size, 3),  # Bottom-right
    ]
    
    for r, c, index in positions:
        if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
            color = grid.values[r][c]
            if color != 0:
                colors[index] = color
            else:
                # Check midpoint
                if index == 0:  # Top-left
                    colors[index] = grid.values[top - 1][left + mid]
                elif index == 1:  # Top-right
                    colors[index] = grid.values[top + mid][left + size]
                elif index == 2:  # Bottom-left
                    colors[index] = grid.values[top + size][left + mid]
                else:  # Bottom-right
                    colors[index] = grid.values[top + mid][left - 1]
    
    return colors
