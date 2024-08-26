from rob_agi.colored_grid import ColoredGrid

def solve_e633a9e5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 5x5 output grid by expanding each cell into a 2x2 area.
    The expansion follows these rules:
    1. The top-left of each 2x2 area always matches the original cell's color.
    2. The top-right color is determined by comparing with the right neighbor (if it exists).
    3. The bottom-left color is determined by comparing with the bottom neighbor (if it exists).
    4. The bottom-right color is determined by comparing with right, bottom, and bottom-right neighbors (if they exist).
    5. In all comparisons, the smaller (or equal) color value is chosen.
    """
    input_values = input_grid.values
    output_values = [[0 for _ in range(5)] for _ in range(5)]

    for r in range(3):
        for c in range(3):
            color = input_values[r][c]
            
            # Set the top-left of the 2x2 area
            output_values[2*r][2*c] = color
            
            # Set the top-right
            output_values[2*r][2*c+1] = min(color, input_values[r][c+1] if c+1 < 3 else color)
            
            # Set the bottom-left
            output_values[2*r+1][2*c] = min(color, input_values[r+1][c] if r+1 < 3 else color)
            
            # Set the bottom-right
            output_values[2*r+1][2*c+1] = min(
                color,
                input_values[r][c+1] if c+1 < 3 else color,
                input_values[r+1][c] if r+1 < 3 else color,
                input_values[r+1][c+1] if r+1 < 3 and c+1 < 3 else color
            )

    return ColoredGrid(values=output_values)
