from rob_agi.colored_grid import ColoredGrid

def solve_67c52801(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving colored cells downward within their columns.
    
    The transformation follows these rules:
    1. The bottom row of the input grid remains unchanged.
    2. Colored cells move vertically downward, maintaining their original column positions and order.
    3. Empty space (black/0) fills from the top down.
    
    The algorithm works as follows:
    1. Initialize the output grid with zeros and copy the bottom row from the input.
    2. Process each column independently, moving non-zero cells downward.
    3. Preserve the relative order of cells within each column.
    4. Return the transformed grid.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified rules.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Copy the bottom row
    output_grid.values[-1] = input_grid.values[-1].copy()
    
    # Process each column
    for col in range(cols):
        non_zero_cells = []
        for row in range(rows - 1):  # Exclude bottom row
            if input_grid.values[row][col] != 0:
                non_zero_cells.append((row, input_grid.values[row][col]))
        
        # Sort non-zero cells by their original row index (bottom to top)
        non_zero_cells.sort(key=lambda x: x[0], reverse=True)
        
        # Place cells in the output grid
        placement_row = rows - 2  # Start just above the bottom row
        for _, value in non_zero_cells:
            if placement_row >= 0:  # Ensure we don't overwrite the bottom row
                output_grid.values[placement_row][col] = value
                placement_row -= 1
    
    return output_grid
