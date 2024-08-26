from rob_agi.colored_grid import ColoredGrid

def solve_e9ac8c9e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by replacing a gray area with an expanded arrangement of surrounding colors.
    
    1. Locate the gray (5) area in the input.
    2. Identify the four colored squares in the quadrants around the gray area.
    3. Create a new grid where the gray area is replaced by an expanded formation of the surrounding colors,
       each color occupying a quarter of the space previously taken by the gray area.
    4. The new formation is placed exactly where the gray area was.
    5. Clear the original positions of the surrounding colors and any remaining gray cells.
    
    The gray area is removed in the output, and the surrounding colored squares are expanded
    to fill the space in a compact, symmetrical arrangement.
    """
    # Find the gray area
    gray_top, gray_left, gray_height, gray_width = find_gray_area(input_grid)
    
    # Find the four colors
    colors = find_colors(input_grid, gray_top, gray_left, gray_height, gray_width)
    
    # Create the output grid as a copy of the input
    output = input_grid.deep_copy()
    
    # Calculate the new block dimensions
    block_height = (gray_height + 1) // 2
    block_width = (gray_width + 1) // 2
    
    # Fill the new formation
    for i in range(2):
        for j in range(2):
            color = colors[i * 2 + j]
            for r in range(block_height):
                for c in range(block_width):
                    if gray_top + i * block_height + r < output.num_rows and gray_left + j * block_width + c < output.num_cols:
                        output.values[gray_top + i * block_height + r][gray_left + j * block_width + c] = color
    
    # Clear original color positions and any remaining gray cells
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] in colors or input_grid.values[r][c] == 5:
                output.values[r][c] = 0
    
    return output

def find_gray_area(grid: ColoredGrid) -> tuple:
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 5:
                height = 1
                width = 1
                while r + height < grid.num_rows and grid.values[r + height][c] == 5:
                    height += 1
                while c + width < grid.num_cols and grid.values[r][c + width] == 5:
                    width += 1
                return r, c, height, width
    return 0, 0, 0, 0  # Default if no gray area found

def find_colors(grid: ColoredGrid, top: int, left: int, height: int, width: int) -> list:
    colors = [0, 0, 0, 0]  # TL, TR, BL, BR
    mid_row = top + height // 2
    mid_col = left + width // 2
    
    # Define quadrant boundaries
    quadrants = [
        (0, top, 0, left, mid_row, mid_col),  # Top-left
        (1, top, mid_col, left + width, mid_row),  # Top-right
        (2, mid_row, 0, left, top + height, mid_col),  # Bottom-left
        (3, mid_row, mid_col, left + width, top + height)  # Bottom-right
    ]
    
    for index, start_row, start_col, end_col, end_row, mid_col in quadrants:
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                    color = grid.values[r][c]
                    if color != 0 and color != 5:
                        colors[index] = color
                        break
            if colors[index] != 0:
                break
    
    return colors
