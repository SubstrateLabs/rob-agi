from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e41c6fd3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying shapes, sorting them by color,
    and arranging them in a single row at the top of the grid.
    
    1. Identifies all shapes in the input grid using connected regions.
    2. Sorts the shapes by color (ascending order).
    3. Places the sorted shapes in a new grid, starting from the top-left corner.
    4. If shapes don't fit in a single row, continues on the next row.
    5. Returns a new ColoredGrid with the arranged shapes.
    """
    # Find all shapes in the grid
    shapes = []
    for color in range(1, 10):  # Colors 1 to 9
        regions = input_grid.find_connected_regions(color)
        for region in regions:
            min_x = min(x for _, x in region)
            max_x = max(x for _, x in region)
            min_y = min(y for y, _ in region)
            max_y = max(y for y, _ in region)
            shapes.append((color, (min_y, min_x, max_y, max_x), region))
    
    # Sort shapes by color
    shapes.sort(key=lambda x: x[0])
    
    # Create a new grid
    rows, cols = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Place shapes in the new grid
    current_col = 0
    current_row = 0
    for color, (min_y, min_x, max_y, max_x), region in shapes:
        shape_height = max_y - min_y + 1
        shape_width = max_x - min_x + 1
        
        # Check if shape fits in current row
        if current_col + shape_width > cols:
            current_col = 0
            current_row += shape_height
        
        # Stop if we run out of rows
        if current_row + shape_height > rows:
            break
        
        # Place the shape
        for y, x in region:
            new_y = current_row + (y - min_y)
            new_x = current_col + (x - min_x)
            new_grid[new_y][new_x] = color
        
        current_col += shape_width
    
    return ColoredGrid(values=new_grid)
