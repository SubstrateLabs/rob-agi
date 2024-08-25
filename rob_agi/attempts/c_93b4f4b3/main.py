from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_93b4f4b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing it to 6 columns while preserving the border and rearranging internal shapes.
    
    1. Extracts the border and internal shapes from the input grid.
    2. Creates a new 6-column grid with the same number of rows as the input.
    3. Places the border in the output grid.
    4. Sorts internal shapes by complexity (area + perimeter).
    5. Places sorted shapes within the border, centered horizontally.
    6. Fills remaining space with the border color.
    
    Returns the transformed ColoredGrid.
    """
    rows, cols = input_grid.get_dimensions()
    border_color = input_grid.values[0][0]
    
    # Extract border
    border = [(r, c) for r in range(rows) for c in range(6) if input_grid.values[r][c] == border_color]
    
    # Extract internal shapes
    internal_shapes = []
    for color in range(10):
        if color != border_color:
            shapes = input_grid.find_connected_regions(color)
            for shape in shapes:
                if any(c >= 6 for _, c in shape):  # Only consider shapes from the right side
                    internal_shapes.append((shape, color))
    
    # Create output grid
    output_grid = ColoredGrid(values=[[border_color for _ in range(6)] for _ in range(rows)])
    
    # Place border
    for r, c in border:
        output_grid.values[r][c] = border_color
    
    # Sort shapes by complexity (area + perimeter)
    def shape_complexity(shape):
        return len(shape[0]) + len(set(shape[0]))
    internal_shapes.sort(key=lambda x: shape_complexity(x), reverse=True)
    
    # Find placement zones
    placement_zones = []
    start = None
    for r in range(rows):
        if output_grid.values[r][1] != border_color:
            if start is None:
                start = r
        elif start is not None:
            placement_zones.append((start, r))
            start = None
    if start is not None:
        placement_zones.append((start, rows))
    
    # Place shapes
    for shape, color in internal_shapes:
        shape_height = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
        shape_width = max(c for _, c in shape) - min(c for _, c in shape) + 1
        
        placed = False
        for start, end in placement_zones:
            if end - start >= shape_height:
                center = (start + end) // 2
                top = center - shape_height // 2
                left = max(1, (6 - shape_width) // 2)
                
                # Check if placement is valid
                if all(output_grid.values[r][c] == border_color 
                       for r in range(top, top + shape_height) 
                       for c in range(left, left + shape_width)):
                    # Place shape
                    min_r = min(r for r, _ in shape)
                    min_c = min(c for _, c in shape)
                    for r, c in shape:
                        output_grid.values[top + r - min_r][left + c - min_c] = color
                    placed = True
                    break
        
        if not placed:
            raise ValueError("Unable to place all shapes")
    
    return output_grid
