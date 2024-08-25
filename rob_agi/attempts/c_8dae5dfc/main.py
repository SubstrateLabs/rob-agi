from rob_agi.colored_grid import ColoredGrid

def solve_8dae5dfc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a specific color mapping rule to each cell.
    
    The color transformation rules are:
    - 0 (black) remains unchanged
    - 1 -> 2 (blue to red)
    - 2 -> 3 (red to green)
    - 3 -> 8 (green to sky)
    - 4 -> 1 (yellow to blue)
    - 6 -> 7 (magenta to orange)
    - 7 -> 4 (orange to yellow)
    - 8 -> 3 (sky to green)
    - Other colors (5 and 9) remain unchanged if present
    
    This process transforms each color while maintaining the overall structure and position of shapes.
    Black (0) cells, representing empty space, are preserved.
    """
    color_mapping = {
        1: 2, 2: 3, 3: 8, 4: 1,
        6: 7, 7: 4, 8: 3
    }

    def transform_color(color: int) -> int:
        return color_mapping.get(color, color)

    rows, cols = len(input_grid.values), len(input_grid.values[0])
    new_values = [[transform_color(input_grid.values[r][c]) for c in range(cols)] for r in range(rows)]
    
    return ColoredGrid(values=new_values)
