from rob_agi.colored_grid import ColoredGrid

def solve_ea9794b1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 10x10 input grid into a 5x5 output grid by processing 2x2 subgrids.
    
    For each 2x2 subgrid:
    1. If all colors in the subgrid are 0 (black), use 0.
    2. Otherwise, choose the color based on the priority order: 3 (green), 9 (brown), 8 (sky), 4 (yellow), 0 (black).
    3. The highest priority non-zero color present in the subgrid is used.
    4. The output color for each 2x2 subgrid is placed in the corresponding position of the 5x5 output grid.
    
    Args:
    input_grid (ColoredGrid): A 10x10 input grid

    Returns:
    ColoredGrid: A 5x5 output grid
    """
    if input_grid.get_dimensions() != (10, 10):
        raise ValueError("Input grid must be 10x10")

    priority_order = [3, 9, 8, 4, 0]

    def process_subgrid(subgrid):
        colors = set(cell for row in subgrid for cell in row)
        if colors == {0}:
            return 0
        for color in priority_order:
            if color in colors:
                return color
        return 0  # This should never happen given the priority list includes 0

    output_values = []
    for i in range(0, 10, 2):
        row = []
        for j in range(0, 10, 2):
            subgrid = [input_grid.values[i+di][j:j+2] for di in range(2)]
            row.append(process_subgrid(subgrid))
        output_values.append(row)

    return ColoredGrid(values=output_values)
