from rob_agi.colored_grid import ColoredGrid

def solve_8dae5dfc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a color inversion rule to each non-black cell.
    
    The color transformation rule is:
    - For non-zero colors: new_color = (10 - original_color) % 10
    - For black (0): remains unchanged
    
    This process effectively "inverts" each color while maintaining the overall structure and position of shapes.
    Black (0) cells, representing empty space, are preserved.
    """
    def transform_color(color: int) -> int:
        return (10 - color) % 10 if color != 0 else 0

    rows, cols = len(input_grid.values), len(input_grid.values[0])
    new_values = [[transform_color(input_grid.values[r][c]) for c in range(cols)] for r in range(rows)]
    
    return ColoredGrid(values=new_values)
