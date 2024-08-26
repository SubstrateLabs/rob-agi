from rob_agi.colored_grid import ColoredGrid

def solve_e9ac8c9e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by replacing a gray area with an expanded arrangement of surrounding colors.
    
    1. Locate the gray (5) area in the input grid.
    2. Identify the four colored squares in the quadrants around the gray area, searching outward from each corner.
    3. Create a new grid where the gray area is replaced by an expanded formation of the surrounding colors,
       each color occupying a quarter of the space previously taken by the gray area.
    4. The new formation is always placed in the center of the grid, regardless of the original gray area's position.
    5. Clear the original positions of the surrounding colors and any remaining gray cells.
    6. Handle cases where the gray area might touch the grid boundaries or have odd dimensions.
    
    The gray area is removed in the output, and the surrounding colored squares are expanded
    to fill the space in a compact, symmetrical arrangement, regardless of their original distances from the gray area.
    The new arrangement is always centered in the grid, maintaining the size of the original gray area.
    """
    # Find the gray area
    gray_top, gray_left, gray_height, gray_width = find_gray_area(input_grid)
    
    # Find the four colors
    colors = find_colors(input_grid, gray_top, gray_left, gray_height, gray_width)
    
    # Create the output grid filled with black (0)
    output = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    
    # Calculate the new block dimensions
    block_height = (gray_height + 1) // 2
    block_width = (gray_width + 1) // 2
    
    # Calculate the center position for the new formation
    center_row = input_grid.num_rows // 2
    center_col = input_grid.num_cols // 2
    new_top = center_row - gray_height // 2
    new_left = center_col - gray_width // 2
    
    # Fill the new formation
    for i in range(2):
        for j in range(2):
            color = colors[i * 2 + j]
            for r in range(block_height):
                for c in range(block_width):
                    output.values[new_top + i * block_height + r][new_left + j * block_width + c] = color
    
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
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # TL, TR, BL, BR
    
    for i, (dr, dc) in enumerate(directions):
        r, c = top + (height - 1) * max(0, dr), left + (width - 1) * max(0, dc)
        while 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
            color = grid.values[r][c]
            if color != 0 and color != 5:
                colors[i] = color
                break
            r += dr
            c += dc
    
    return colors
