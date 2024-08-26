from rob_agi.colored_grid import ColoredGrid

def solve_40f6cd08(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by replicating the top-left quadrant to all four quadrants of the grid.
    
    1. Extract the top-left quadrant
    2. Create a new output grid
    3. Replicate the top-left quadrant to all four quadrants, adjusting size if necessary
    4. Preserve color patterns and structures within each quadrant
    5. Fill any remaining space with black (0)
    
    Returns a new 30x30 ColoredGrid with the transformed pattern.
    """
    # Extract the top-left quadrant (15x15)
    top_left = input_grid.extract_subgrid(0, 0, 15, 15)
    
    # Create a new output grid
    output = ColoredGrid(values=[[0 for _ in range(30)] for _ in range(30)])
    
    # Function to copy and scale a subgrid to a target area
    def copy_and_scale(subgrid, target_top, target_left, target_height, target_width):
        src_height, src_width = subgrid.get_dimensions()
        for i in range(target_height):
            for j in range(target_width):
                src_i = min(i * src_height // target_height, src_height - 1)
                src_j = min(j * src_width // target_width, src_width - 1)
                output.values[target_top + i][target_left + j] = subgrid.values[src_i][src_j]
    
    # Replicate to all four quadrants
    copy_and_scale(top_left, 0, 0, 15, 15)  # Top-left (original)
    copy_and_scale(top_left, 0, 15, 15, 15)  # Top-right
    copy_and_scale(top_left, 15, 0, 15, 15)  # Bottom-left
    copy_and_scale(top_left, 15, 15, 15, 15)  # Bottom-right
    
    return output
