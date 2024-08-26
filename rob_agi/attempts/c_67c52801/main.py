from rob_agi.colored_grid import ColoredGrid

def solve_67c52801(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving colored cells downward and to the left.
    
    The transformation follows these rules:
    1. The bottom row of the input grid remains unchanged.
    2. Colored cells move downward and to the left, maintaining their relative order.
    3. Empty space (black/0) fills from the top and right.
    
    The algorithm works as follows:
    1. Initialize the output grid with zeros and copy the bottom row from the input.
    2. Process rows from bottom to top, moving non-zero cells downward and to the left.
    3. Preserve the relative order of cells within each row.
    4. Fill any remaining space with zeros.
    5. Return the transformed grid.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified rules.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Copy the bottom row
    output_grid.values[-1] = input_grid.values[-1].copy()
    
    # Process rows from bottom to top
    placement_row = rows - 2  # Start just above the bottom row
    placement_col = 0
    
    for row in range(rows - 2, -1, -1):  # Exclude bottom row, process from second-to-last to top
        non_zero_cells = [cell for cell in input_grid.values[row] if cell != 0]
        
        for value in non_zero_cells:
            if placement_col < cols:
                output_grid.values[placement_row][placement_col] = value
                placement_col += 1
            else:
                placement_row -= 1
                placement_col = 0
                if placement_row >= 0:
                    output_grid.values[placement_row][placement_col] = value
                    placement_col += 1
                else:
                    break  # No more space to place cells
        
        if placement_col == cols:
            placement_row -= 1
            placement_col = 0
    
    return output_grid
