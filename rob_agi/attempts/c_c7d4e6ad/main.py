from rob_agi.colored_grid import ColoredGrid

def solve_c7d4e6ad(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing gray squares (color 5) with the leftmost
    non-black, non-gray color in each row. Other colors remain unchanged.
    
    The transformation process:
    1. For each row, find the leftmost non-black (0), non-gray (5) color.
    2. Replace all gray squares in the row with this color.
    3. Leave other colors unchanged.
    4. If no such color is found, the row remains unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    def transform_row(row):
        leftmost_color = next((color for color in row if color not in {0, 5}), None)
        return [leftmost_color if cell == 5 and leftmost_color is not None else cell for cell in row]

    transformed_values = [transform_row(row) for row in input_grid.values]
    return ColoredGrid(values=transformed_values)
