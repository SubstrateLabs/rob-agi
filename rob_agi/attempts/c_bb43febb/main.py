from rob_agi.colored_grid import ColoredGrid

def solve_bb43febb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the bb43febb challenge by identifying rectangular shapes formed by color 5,
    preserving their outer border, and filling their interior with color 2.
    
    The function detects rectangles in the grid, then fills the interior of each
    rectangle with color 2 while keeping the border as color 5.
    """
    def find_rectangles(grid):
        rectangles = []
        rows, cols = grid.get_dimensions()
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 5 and not visited[r][c]:
                    # Find the bottom-right corner of the rectangle
                    br_row, br_col = r, c
                    while br_row + 1 < rows and grid.get_cell(br_row + 1, c) == 5:
                        br_row += 1
                    while br_col + 1 < cols and grid.get_cell(r, br_col + 1) == 5:
                        br_col += 1
                    
                    # Mark all cells in the rectangle as visited
                    for i in range(r, br_row + 1):
                        for j in range(c, br_col + 1):
                            visited[i][j] = True
                    
                    rectangles.append((r, c, br_row, br_col))
        
        return rectangles

    def fill_rectangle(grid, rect):
        top, left, bottom, right = rect
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                grid.set_cell(r, c, 2)

    # Create a deep copy of the input grid
    result = input_grid.deep_copy()
    
    # Find all rectangles in the grid
    rectangles = find_rectangles(result)
    
    # Fill the interior of each rectangle with color 2
    for rect in rectangles:
        fill_rectangle(result, rect)
    
    return result
