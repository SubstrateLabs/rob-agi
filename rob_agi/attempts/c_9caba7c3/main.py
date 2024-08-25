from rob_agi.colored_grid import ColoredGrid

def solve_9caba7c3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Changes each red (2) square to yellow (4).
    2. Changes adjacent gray (5) or red (2) squares to orange (7).
    3. Preserves the original state of other colors.
    4. Applies transformations simultaneously by referring to the original grid state.
    """
    # Create a deep copy of the input grid
    original_grid = input_grid.deep_copy()
    
    # Find all red squares
    red_squares = []
    for row in range(len(original_grid.values)):
        for col in range(len(original_grid.values[0])):
            if original_grid.values[row][col] == 2:
                red_squares.append((row, col))
    
    # Create a second copy for transformations
    transformed_grid = original_grid.deep_copy()
    
    # Apply transformations
    for row, col in red_squares:
        # Change red to yellow
        transformed_grid.values[row][col] = 4
        
        # Check and transform adjacent squares
        adjacent = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        for adj_row, adj_col in adjacent:
            if 0 <= adj_row < len(original_grid.values) and 0 <= adj_col < len(original_grid.values[0]):
                if original_grid.values[adj_row][adj_col] in [2, 5]:
                    transformed_grid.values[adj_row][adj_col] = 7
    
    # Return the transformed grid
    return transformed_grid
