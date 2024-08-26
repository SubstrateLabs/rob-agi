from rob_agi.colored_grid import ColoredGrid

def solve_d47aa2ff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 10x21 input grid into a 10x10 output grid by:
    1. Extracting the left 10x10 portion of the input grid.
    2. Adding one blue (1) dot and one red (2) dot to the right half.
    3. Placing the blue dot in the first available empty space from top-right.
    4. Placing the red dot in the next available empty space after the blue dot.

    Args:
    input_grid (ColoredGrid): A 10x21 grid with a central gray (5) line.

    Returns:
    ColoredGrid: A 10x10 grid with the transformation applied.
    """
    # Step 1: Extract the left 10x10 portion of the input grid
    output_grid = ColoredGrid(values=[row[:10] for row in input_grid.values[:10]])
    
    # Step 2 & 3: Add blue dot to the first available space in the right half
    blue_placed = False
    for i in range(10):
        for j in range(5, 10):
            if output_grid.values[i][j] == 0:
                output_grid.values[i][j] = 1  # Place blue dot
                blue_placed = True
                break
        if blue_placed:
            break
    
    # Step 4: Add red dot to the next available space after the blue dot
    red_placed = False
    for i in range(10):
        for j in range(5, 10):
            if output_grid.values[i][j] == 0:
                output_grid.values[i][j] = 2  # Place red dot
                red_placed = True
                break
        if red_placed:
            break
    
    return output_grid
