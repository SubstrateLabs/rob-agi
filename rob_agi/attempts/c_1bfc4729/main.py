from rob_agi.colored_grid import ColoredGrid

def solve_1bfc4729(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying a pattern based on two key colors.
    
    The function identifies two key colors in the input grid and applies a specific pattern:
    - The top color fills the top three rows and side columns of the top half.
    - The bottom color fills the bottom three rows and side columns of the bottom half.
    - The middle section (rows 4-6) has only the side columns filled with the bottom color.
    - The rows corresponding to the original positions of the key colors are fully filled.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the applied pattern.
    """
    def find_key_colors(grid):
        colors = []
        for i, row in enumerate(grid.values):
            for j, cell in enumerate(row):
                if cell != 0 and len(colors) < 2:
                    colors.append((cell, i))
            if len(colors) == 2:
                break
        return colors

    def apply_pattern(grid, color, start_row, end_row):
        for i in range(start_row, end_row):
            grid[i][0] = grid[i][-1] = color
        grid[start_row] = grid[start_row + 2] = [color] * 10

    colors = find_key_colors(input_grid)
    output = [[0 for _ in range(10)] for _ in range(10)]

    if len(colors) >= 1:
        top_color, top_row = colors[0]
        apply_pattern(output, top_color, 0, 5)
        output[top_row] = [top_color] * 10

    if len(colors) == 2:
        bottom_color, bottom_row = colors[1]
        apply_pattern(output, bottom_color, 5, 10)
        output[bottom_row] = [bottom_color] * 10

    return ColoredGrid(values=output)
