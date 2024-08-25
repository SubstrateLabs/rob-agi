from rob_agi.colored_grid import ColoredGrid

def solve_72207abc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a pattern to the middle row.
    
    The pattern:
    1. Extracts the initial sequence of non-zero colors from the middle row.
    2. Repeats this sequence across the row with increasing spacing between colors.
    3. The spacing starts at 1 and increases by 2 after each complete sequence.
    4. The top and bottom rows remain unchanged (all zeros).
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid with the pattern applied.
    """
    # Extract the initial sequence
    initial_sequence = []
    for color in input_grid.values[1]:  # Middle row
        if color != 0:
            initial_sequence.append(color)
        elif initial_sequence:  # If we've found at least one non-zero color
            break
    
    # Create a new grid
    new_grid = input_grid.deep_copy()
    
    # Fill the middle row (initial part)
    current_position = 0
    for color in initial_sequence:
        new_grid.values[1][current_position] = color
        current_position += 1
    spacing = 1
    
    # Continue filling the middle row
    width = len(input_grid.values[1])
    while current_position < width:
        for color in initial_sequence:
            # Add spacing
            for _ in range(spacing):
                if current_position < width:
                    new_grid.values[1][current_position] = 0
                    current_position += 1
                else:
                    break
            
            # Add color
            if current_position < width:
                new_grid.values[1][current_position] = color
                current_position += 1
            else:
                break
        
        spacing += 2
    
    return new_grid
