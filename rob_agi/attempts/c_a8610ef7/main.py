from rob_agi.colored_grid import ColoredGrid

def create_checkerboard(rows, cols):
    return [[2 if (r + c) % 2 == 0 else 5 for c in range(cols)] for r in range(rows)]

def count_adjacent_colors(grid, row, col):
    counts = {2: 0, 5: 0}
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] in [2, 5]:
                counts[grid[r][c]] += 1
    return counts

def get_checkerboard_color(row, col):
    return 2 if (row + col) % 2 == 0 else 5

def solve_a8610ef7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by replacing sky blue (8) regions with red (2) and gray (5) colors.
    The algorithm uses a virtual checkerboard pattern as a starting point, then adjusts based on
    adjacent colors and local context. It ensures a consistent alternating pattern while
    respecting the existing structure of non-8 colors in the input grid.
    """
    grid_2d = input_grid.values
    rows, cols = len(grid_2d), len(grid_2d[0])
    
    checkerboard = create_checkerboard(rows, cols)
    output_grid = [[cell if cell != 8 else 0 for cell in row] for row in grid_2d]
    
    # First pass: color most 8s
    for r in range(rows):
        for c in range(cols):
            if grid_2d[r][c] == 8:
                counts = count_adjacent_colors(output_grid, r, c)
                if counts[2] > counts[5]:
                    output_grid[r][c] = 5
                elif counts[5] > counts[2]:
                    output_grid[r][c] = 2
                else:
                    checkerboard_color = get_checkerboard_color(r, c)
                    if checkerboard_color not in counts:
                        output_grid[r][c] = checkerboard_color
                    else:
                        output_grid[r][c] = 7 - checkerboard_color  # 7 - 2 = 5, 7 - 5 = 2
    
    # Second pass: handle any remaining uncolored cells
    for r in range(rows):
        for c in range(cols):
            if output_grid[r][c] == 0:
                counts = count_adjacent_colors(output_grid, r, c)
                if counts[2] > counts[5]:
                    output_grid[r][c] = 5
                elif counts[5] > counts[2]:
                    output_grid[r][c] = 2
                else:
                    output_grid[r][c] = get_checkerboard_color(r, c)
    
    return ColoredGrid(values=output_grid)
