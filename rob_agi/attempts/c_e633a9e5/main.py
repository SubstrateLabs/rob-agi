from rob_agi.colored_grid import ColoredGrid

def solve_e633a9e5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 5x5 output grid by expanding each cell into a 2x2 area.
    The expansion follows these rules:
    1. The top-left and bottom-right of each 2x2 area always match the original cell's color.
    2. The top-right color is determined by comparing with the right neighbor (if it exists).
    3. The bottom-left color is determined by comparing with the bottom neighbor (if it exists).
    4. In comparisons, the smaller (or equal) color value is chosen.
    """
    input_values = input_grid.values
    output_values = [[0 for _ in range(5)] for _ in range(5)]

    for r in range(3):
        for c in range(3):
            color = input_values[r][c]
            
            # Always set the top-left and bottom-right of the 2x2 area
            output_values[2*r][2*c] = color
            output_values[2*r+1][2*c+1] = color
            
            # Handle top-right (compare with right neighbor if it exists)
            if c < 2:
                right_color = input_values[r][c+1]
                output_values[2*r][2*c+1] = min(color, right_color)
            else:
                output_values[2*r][2*c+1] = color
            
            # Handle bottom-left (compare with bottom neighbor if it exists)
            if r < 2:
                bottom_color = input_values[r+1][c]
                output_values[2*r+1][2*c] = min(color, bottom_color)
            else:
                output_values[2*r+1][2*c] = color

    return ColoredGrid(values=output_values)
