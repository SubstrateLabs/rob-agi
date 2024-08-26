from rob_agi.colored_grid import ColoredGrid

def solve_48131b3c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by doubling its size in both dimensions and creating a tiled pattern.
    The transformation follows these steps:
    1. Create a new grid twice the size of the input in both dimensions.
    2. For each 2x2 block in the output:
       - Top-left: Copy from input
       - Top-right: Take from right in input (wrap if needed)
       - Bottom-left: Take from below in input (wrap if needed)
       - Bottom-right: Take from diagonally down-right in input (wrap if needed)
    3. Repeat this pattern to fill the entire output grid, creating a seamless tiled pattern.
    """
    input_height, input_width = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(input_width*2)] for _ in range(input_height*2)])
    
    for i in range(input_height):
        for j in range(input_width):
            # Fill top-left quadrant
            output_grid.values[i][j] = input_grid.values[i][j]
            # Fill top-right quadrant
            output_grid.values[i][j + input_width] = input_grid.values[i][(j+1) % input_width]
            # Fill bottom-left quadrant
            output_grid.values[i + input_height][j] = input_grid.values[(i+1) % input_height][j]
            # Fill bottom-right quadrant
            output_grid.values[i + input_height][j + input_width] = input_grid.values[(i+1) % input_height][(j+1) % input_width]
    
    return output_grid
