from rob_agi.colored_grid import ColoredGrid

def solve_48131b3c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by doubling its size in both dimensions and creating a specific pattern.
    The transformation follows these steps:
    1. Create a new grid twice the size of the input in both dimensions.
    2. For each cell in the input grid, create a 2x2 block in the output grid:
       - Top-left: Copy the value from the input cell
       - Top-right: Take the value from the cell to the right in the input (wrap if needed)
       - Bottom-left: Take the value from the cell below in the input (wrap if needed)
       - Bottom-right: Copy the value from the input cell
    3. This creates a pattern where every other row and column is stretched, resulting in a unique tiled effect.
    """
    input_height, input_width = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(input_width*2)] for _ in range(input_height*2)])
    
    for i in range(input_height):
        for j in range(input_width):
            current_value = input_grid.values[i][j]
            right_value = input_grid.values[i][(j+1) % input_width]
            bottom_value = input_grid.values[(i+1) % input_height][j]
            
            # Fill 2x2 block in output grid
            output_grid.values[2*i][2*j] = current_value       # Top-left
            output_grid.values[2*i][2*j + 1] = right_value     # Top-right
            output_grid.values[2*i + 1][2*j] = bottom_value    # Bottom-left
            output_grid.values[2*i + 1][2*j + 1] = current_value  # Bottom-right
    
    return output_grid
