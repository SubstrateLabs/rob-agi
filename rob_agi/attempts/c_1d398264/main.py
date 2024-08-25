from rob_agi.colored_grid import ColoredGrid

def is_valid_position(row, col, grid):
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])

def expand(start_row, start_col, direction, color, grid):
    current_row, current_col = start_row, start_col
    while is_valid_position(current_row, current_col, grid):
        if grid[current_row][current_col] != 0 and (current_row, current_col) != (start_row, start_col):
            break
        grid[current_row][current_col] = color
        current_row += direction[0]
        current_col += direction[1]

def solve_1d398264(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding non-black cells according to color-specific rules.
    Each color has a primary expansion direction, and some have a secondary direction.
    The expansion continues until it hits an edge or a non-black cell.
    The original non-black cells are preserved, and expansions are applied in a specific order.
    """
    grid = input_grid.deep_copy()
    expansion_rules = {
        1: ((-1, 1), None),  # Blue: diagonal up-right
        2: ((0, 1), (1, 0)),  # Red: horizontal, then vertical at ends
        3: ((1, 1), (-1, -1)),  # Green: diagonal down-right and up-left
        4: ((0, 1), None),  # Yellow: horizontal
        5: ((0, 0), None),  # Gray: no expansion
        6: ((-1, -1), None),  # Magenta: diagonal up-left
        7: ((1, 1), None),  # Orange: diagonal down-right
        8: ((1, 0), (0, 1)),  # Sky Blue: vertical down, then horizontal at bottom
    }
    
    non_black_cells = [(r, c, grid[r][c]) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != 0]
    
    for row, col, color in non_black_cells:
        primary, secondary = expansion_rules[color]
        expand(row, col, primary, color, grid)
        if secondary:
            end_row, end_col = row, col
            while is_valid_position(end_row + primary[0], end_col + primary[1], grid):
                end_row += primary[0]
                end_col += primary[1]
            expand(end_row, end_col, secondary, color, grid)
    
    return ColoredGrid(values=grid)
