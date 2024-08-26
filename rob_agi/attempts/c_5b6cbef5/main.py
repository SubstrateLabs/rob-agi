from rob_agi.colored_grid import ColoredGrid

def solve_5b6cbef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x4 input grid into a 16x16 output grid by mirroring the input pattern.
    
    The solution involves the following steps:
    1. Copy the input to the top-left quadrant (0-3, 0-3)
    2. Mirror horizontally to fill the top-right quadrant (0-3, 12-15)
    3. Mirror vertically to fill the bottom-left quadrant (12-15, 0-3)
    4. Mirror both horizontally and vertically to fill the bottom-right quadrant (12-15, 12-15)
    
    This mirroring process creates a symmetrical pattern that expands the input across the larger grid.
    """
    # Initialize a new 16x16 ColoredGrid
    new_grid = ColoredGrid(values=[[0 for _ in range(16)] for _ in range(16)])
    
    # Apply the mirroring logic
    for i in range(4):
        for j in range(4):
            value = input_grid.values[i][j]
            
            # Top-left quadrant (direct copy)
            new_grid.values[i][j] = value
            
            # Top-right quadrant (horizontal mirror)
            new_grid.values[i][15-j] = value
            
            # Bottom-left quadrant (vertical mirror)
            new_grid.values[15-i][j] = value
            
            # Bottom-right quadrant (both horizontal and vertical mirror)
            new_grid.values[15-i][15-j] = value
    
    return new_grid
