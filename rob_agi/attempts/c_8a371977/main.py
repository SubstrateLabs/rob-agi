from rob_agi.colored_grid import ColoredGrid

def solve_8a371977(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by coloring regions in a concentric pattern.
    
    The solution works as follows:
    1. Identifies the grid structure (1x1 checkered or 3x3 regions).
    2. Calculates the layer number for each cell based on its distance from the edge.
    3. Colors the regions based on their layer number and grid structure:
       - For 1x1 checkered pattern:
         * Outermost layer is red (2).
         * Second layer is red (2) for corner cells, green (3) for others.
         * Inner layers alternate between green (3) and red (2).
       - For 3x3 region pattern:
         * Layers alternate between red (2) and green (3).
    4. Handles special cases for the corners in checkered grids.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    def is_checkered_pattern():
        return any(input_grid.values[r][c] == 0 for r in range(rows) for c in range(cols))
    
    checkered = is_checkered_pattern()
    
    def get_layer(r, c):
        return min(r, c, rows-1-r, cols-1-c)
    
    def get_color(layer, r, c):
        if checkered:
            if layer == 0:
                return 2
            elif layer == 1:
                return 2 if (r == 1 and c == 1) or (r == 1 and c == cols-2) or (r == rows-2 and c == 1) or (r == rows-2 and c == cols-2) else 3
            else:
                return 3 if layer % 2 == 1 else 2
        else:
            return 2 if layer % 2 == 0 else 3
    
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 0:
                layer = get_layer(r, c)
                output_grid.values[r][c] = get_color(layer, r, c)
    
    return output_grid
