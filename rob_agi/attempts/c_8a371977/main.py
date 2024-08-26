from rob_agi.colored_grid import ColoredGrid

def solve_8a371977(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by coloring regions in a concentric pattern.
    
    The solution works as follows:
    1. Identifies the grid structure (1x1 checkered or larger regions).
    2. Calculates the distance of each cell from the edge.
    3. Colors the regions based on their distance and grid structure:
       - Outermost layer is always red (2).
       - For larger regions, alternates between red (2) and green (3).
       - For 1x1 checkered, inner cells are green (3) except for the four innermost corners.
    4. Handles special cases for center regions in different grid sizes.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    def is_checkered_pattern():
        center_r, center_c = rows // 2, cols // 2
        return (input_grid.values[center_r][center_c] == 1 and
                input_grid.values[center_r][center_c+1] == 0 and
                input_grid.values[center_r+1][center_c] == 0 and
                input_grid.values[center_r+1][center_c+1] == 1)
    
    checkered = is_checkered_pattern()
    
    def get_distance(r, c):
        return min(r, c, rows-1-r, cols-1-c)
    
    def get_color(r, c, dist):
        if dist == 0:
            return 2
        if checkered:
            if dist == 1:
                return 2
            if (r in (1, rows-2) and c in (1, cols-2)) or (r in (2, rows-3) and c in (2, cols-3)):
                return 2
            return 3
        else:
            return 2 if dist % 2 == 0 else 3
    
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 0:
                dist = get_distance(r, c)
                output_grid.values[r][c] = get_color(r, c, dist)
    
    return output_grid
