from rob_agi.colored_grid import ColoredGrid

def solve_50cb2852(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 50cb2852 challenge by identifying rectangular regions of colors 1, 2, and 3,
    then filling their interiors with color 8 while keeping the outer borders intact.
    
    The function processes all rectangles in the grid, regardless of their position or size,
    and handles potential edge cases such as single-cell rectangles or rectangles touching grid borders.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with filled rectangles.
    """
    def find_rectangles(grid):
        rectangles = []
        height, width = grid.get_dimensions()
        visited = [[False for _ in range(width)] for _ in range(height)]
        
        for i in range(height):
            for j in range(width):
                if not visited[i][j] and grid.get_cell(i, j) in [1, 2, 3]:
                    color = grid.get_cell(i, j)
                    top, left = i, j
                    bottom, right = i, j
                    
                    # Find bottom-right corner
                    while bottom + 1 < height and grid.get_cell(bottom + 1, j) == color:
                        bottom += 1
                    while right + 1 < width and grid.get_cell(i, right + 1) == color:
                        right += 1
                    
                    # Mark as visited
                    for r in range(top, bottom + 1):
                        for c in range(left, right + 1):
                            visited[r][c] = True
                    
                    rectangles.append((top, left, bottom, right, color))
        
        return rectangles

    def fill_rectangle(grid, rect):
        top, left, bottom, right, color = rect
        for i in range(top + 1, bottom):
            for j in range(left + 1, right):
                grid.set_cell(i, j, 8)
        return grid

    output = input_grid.deep_copy()
    rectangles = find_rectangles(output)
    
    for rect in rectangles:
        output = fill_rectangle(output, rect)
    
    return output
