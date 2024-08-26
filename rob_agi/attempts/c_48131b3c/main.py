from rob_agi.colored_grid import ColoredGrid

def solve_48131b3c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by doubling its size in both dimensions.
    Each cell in the input grid is expanded into a 2x2 block in the output grid.
    The expansion follows a pattern where:
    - Top-left cell of the block is the same as the input cell
    - Top-right cell takes the color from the right (wrapping around if needed)
    - Bottom-left cell takes the color from below (wrapping around if needed)
    - Bottom-right cell takes the color from diagonally down-right (wrapping around if needed)
    This creates a seamless tiling pattern that's twice the size of the input in both dimensions.
    """
    input_height, input_width = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(input_width*2)] for _ in range(input_height*2)])
    
    for i in range(input_height):
        for j in range(input_width):
            # Top-left: same as input
            output_grid.values[i*2][j*2] = input_grid.values[i][j]
            
            # Top-right: color from the right (wrap around if needed)
            output_grid.values[i*2][j*2 + 1] = input_grid.values[i][(j+1) % input_width]
            
            # Bottom-left: color from below (wrap around if needed)
            output_grid.values[i*2 + 1][j*2] = input_grid.values[(i+1) % input_height][j]
            
            # Bottom-right: color from diagonally down-right (wrap around if needed)
            output_grid.values[i*2 + 1][j*2 + 1] = input_grid.values[(i+1) % input_height][(j+1) % input_width]
    
    return output_grid
