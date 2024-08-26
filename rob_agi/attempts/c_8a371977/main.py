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
         * Inner layers are all green (3).
       - For 3x3 region pattern:
         * Outermost layer is red (2).
         * All inner layers are green (3).
    4. Keeps blue (1) cells unchanged.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    def is_checkered_pattern():
        return any(input_grid.values[r][c] == 0 for r in range(rows) for c in range(cols))
    
    checkered = is_checkered_pattern()
    
    def get_layer(r, c):
        return min(r, c, rows-1-r, cols-1-c)
    
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 1:
                continue  # Keep blue cells unchanged
            
            layer = get_layer(r, c)
            
            if layer == 0:
                output_grid.values[r][c] = 2  # Outermost layer is always red
            elif checkered and layer == 1 and (r, c) in [(1,1), (1,cols-2), (rows-2,1), (rows-2,cols-2)]:
                output_grid.values[r][c] = 2  # Corner cells of second layer in checkered pattern
            else:
                output_grid.values[r][c] = 3  # All other cells are green
    
    return output_grid
