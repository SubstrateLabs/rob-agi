from rob_agi.colored_grid import ColoredGrid

def solve_2697da3f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger, symmetrical pattern.
    
    The transformation involves:
    1. Expanding the input grid to a larger size (2n-1 where n is the input dimension).
    2. Creating a symmetrical pattern by rotating and mirroring the input.
    3. Extending patterns to the edges and corners of the output grid.
    4. Ensuring four-fold rotational symmetry in the final output.
    """
    input_rows, input_cols = input_grid.get_dimensions()
    output_size = max(input_rows, input_cols) * 2 - 1
    output_grid = [[0 for _ in range(output_size)] for _ in range(output_size)]
    
    # Find the center of both input and output grids
    input_center = (input_rows // 2, input_cols // 2)
    output_center = (output_size // 2, output_size // 2)
    
    # Transform the input to create one quadrant of the output
    for r in range(input_rows):
        for c in range(input_cols):
            if input_grid.values[r][c] != 0:
                # Calculate relative position
                rel_r, rel_c = r - input_center[0], c - input_center[1]
                # Place in output grid
                output_grid[output_center[0] + rel_r][output_center[1] + rel_c] = input_grid.values[r][c]
                
                # Extend to edges if on the edge of input
                if r == 0 or r == input_rows - 1 or c == 0 or c == input_cols - 1:
                    if rel_r == 0:
                        for i in range(output_center[1] + rel_c, output_size):
                            output_grid[output_center[0]][i] = input_grid.values[r][c]
                    if rel_c == 0:
                        for i in range(output_center[0] + rel_r, output_size):
                            output_grid[i][output_center[1]] = input_grid.values[r][c]
    
    # Apply rotational symmetry
    for r in range(output_size):
        for c in range(output_size):
            if output_grid[r][c] != 0:
                # Rotate 90 degrees
                output_grid[c][output_size-1-r] = output_grid[r][c]
                # Rotate 180 degrees
                output_grid[output_size-1-r][output_size-1-c] = output_grid[r][c]
                # Rotate 270 degrees
                output_grid[output_size-1-c][r] = output_grid[r][c]
    
    # Fill corners based on edge patterns
    if output_grid[output_center[0]][output_center[1]+1] != 0:
        corner_color = output_grid[output_center[0]][output_center[1]+1]
        output_grid[0][0] = corner_color
        output_grid[0][output_size-1] = corner_color
        output_grid[output_size-1][0] = corner_color
        output_grid[output_size-1][output_size-1] = corner_color
    
    return ColoredGrid(values=output_grid)
