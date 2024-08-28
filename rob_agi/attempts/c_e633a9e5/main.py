from rob_agi.colored_grid import ColoredGrid

def solve_e633a9e5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 5x5 output grid by expanding each cell into a 2x2 area.
    The expansion follows these rules:
    1. Each input cell is expanded into a 2x2 block in the output grid.
    2. The top-left cell of each 2x2 block is always the same as the corresponding input cell.
    3. For other cells in the 2x2 block:
       - Top-right: minimum of left neighbor and right input cell (if exists)
       - Bottom-left: minimum of top neighbor and bottom input cell (if exists)
       - Bottom-right: minimum of top-left, top-right, and bottom-left neighbors
    4. Edge cases are handled by using the input cell's value when there's no neighbor.
    """
    input_values = input_grid.values
    output_values = [[None for _ in range(5)] for _ in range(5)]

    for r in range(3):
        for c in range(3):
            # Top-left of 2x2 block
            output_values[2*r][2*c] = input_values[r][c]
            
            # Top-right of 2x2 block
            if c < 2:
                output_values[2*r][2*c+1] = min(input_values[r][c], input_values[r][c+1])
            else:
                output_values[2*r][2*c+1] = input_values[r][c]
            
            # Bottom-left of 2x2 block
            if r < 2:
                output_values[2*r+1][2*c] = min(input_values[r][c], input_values[r+1][c])
            else:
                output_values[2*r+1][2*c] = input_values[r][c]
            
            # Bottom-right of 2x2 block
            output_values[2*r+1][2*c+1] = min(input_values[r][c], 
                                              output_values[2*r][2*c+1], 
                                              output_values[2*r+1][2*c])

    return ColoredGrid(values=output_values)
