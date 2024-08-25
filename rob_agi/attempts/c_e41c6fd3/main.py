from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_e41c6fd3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying shapes, grouping them by color family,
    sorting within each family, and arranging them efficiently in the upper part of the grid.
    
    1. Identifies all shapes in the input grid using connected regions.
    2. Groups shapes into color families: Cool (1, 8), Warm (2, 4), Other (3, 5, 6, 7, 9).
    3. Sorts shapes within each family by color number.
    4. Alternates between color families when arranging shapes.
    5. Calculates total shape area to determine vertical positioning.
    6. Places shapes efficiently, preserving their internal structure and spacing.
    7. Adjusts vertical positioning based on total shape area.
    8. Maintains two rows of black space at the top when vertical centering is not possible.
    9. Returns a new ColoredGrid with the arranged shapes.
    """
    # Find all shapes in the grid
    shapes = []
    for color in range(1, 10):  # Colors 1 to 9
        regions = input_grid.find_connected_regions(color)
        for region in regions:
            min_y = min(y for y, _ in region)
            max_y = max(y for y, _ in region)
            min_x = min(x for _, x in region)
            max_x = max(x for _, x in region)
            shapes.append((color, (min_y, min_x, max_y, max_x), region))
    
    # Group shapes by color family and sort
    color_families: Dict[str, List[int]] = {
        "Cool": [1, 8],
        "Warm": [2, 4],
        "Other": [3, 5, 6, 7, 9]
    }
    family_shapes = {family: sorted([shape for shape in shapes if shape[0] in colors], key=lambda x: x[0])
                     for family, colors in color_families.items()}
    
    # Interleave shapes from different families
    grouped_shapes = []
    max_shapes = max(len(shapes) for shapes in family_shapes.values())
    for i in range(max_shapes):
        for family in ["Cool", "Warm", "Other"]:
            if i < len(family_shapes[family]):
                grouped_shapes.append(family_shapes[family][i])
    
    # Calculate total shape area
    total_area = sum((max_y - min_y + 1) * (max_x - min_x + 1) for _, (min_y, min_x, max_y, max_x), _ in shapes)
    
    # Create a new grid
    rows, cols = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Determine starting row
    vertical_centering = total_area <= (rows * cols) // 2
    start_row = 2 if not vertical_centering else (rows - (total_area // cols)) // 2
    
    # Place shapes in the new grid
    current_col = 0
    current_row = start_row
    row_height = 0
    for color, (min_y, min_x, max_y, max_x), region in grouped_shapes:
        shape_height = max_y - min_y + 1
        shape_width = max_x - min_x + 1
        
        # Check if shape fits in current row
        if current_col + shape_width + 1 > cols:
            current_row += row_height + 1
            current_col = 0
            row_height = 0
        
        # Stop if we run out of rows
        if current_row + shape_height > rows:
            break
        
        # Place the shape
        for y, x in region:
            new_y = current_row + (y - min_y)
            new_x = current_col + (x - min_x)
            new_grid[new_y][new_x] = color
        
        current_col += shape_width + 1
        row_height = max(row_height, shape_height)
    
    # Adjust vertical position if needed
    if vertical_centering:
        used_rows = max(y for row in new_grid for y, cell in enumerate(row) if cell != 0) + 1
        shift = (rows - used_rows) // 2 - start_row
        if shift > 0:
            new_grid = [[0] * cols] * shift + new_grid[:-shift]
    
    return ColoredGrid(values=new_grid)
