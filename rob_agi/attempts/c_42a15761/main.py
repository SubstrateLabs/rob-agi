from rob_agi.colored_grid import ColoredGrid

def solve_42a15761(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a grid of 'E' shapes by fixing inconsistencies in the middle and bottom bars.
    The transformation follows these rules:
    1. Top bars and vertical segments of 'E's are always full.
    2. Middle bars alternate between full and partial within each column.
    3. Bottom bars are full when the middle bar is partial, and partial when the middle bar is full.
    4. The direction of partial bars (left or right) is consistent within a column but may vary between columns.
    5. Black vertical separating lines remain unchanged.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()
    
    def get_e_height() -> int:
        for r in range(1, rows):
            if new_grid.values[r] == new_grid.values[0]:
                return r
        return rows

    def get_num_columns() -> int:
        return sum(1 for val in new_grid.values[0] if val == 0) + 1

    def get_partial_bar_direction(col: int) -> str:
        e_start = col * 4
        middle_row = e_height // 2
        if new_grid.values[middle_row][e_start + 1] == 2:
            return 'left'
        return 'right'

    def is_full_middle_bar(e_index: int) -> bool:
        return e_index % 2 == 0

    e_height = get_e_height()
    num_columns = get_num_columns()
    
    for col in range(num_columns):
        direction = get_partial_bar_direction(col)
        for e in range(rows // e_height):
            e_start = col * 4
            e_top = e * e_height
            
            # Fix top bar
            new_grid.values[e_top][e_start:e_start+3] = [2, 2, 2]
            
            # Fix vertical segments
            for r in range(e_top, e_top + e_height):
                new_grid.values[r][e_start] = 2
                new_grid.values[r][e_start+2] = 2
            
            # Fix middle bar
            middle_row = e_top + e_height // 2
            if is_full_middle_bar(e):
                new_grid.values[middle_row][e_start:e_start+3] = [2, 2, 2]
            else:
                if direction == 'left':
                    new_grid.values[middle_row][e_start:e_start+3] = [2, 0, 0]
                else:
                    new_grid.values[middle_row][e_start:e_start+3] = [0, 0, 2]
            
            # Fix bottom bar
            bottom_row = e_top + e_height - 1
            if is_full_middle_bar(e):
                if direction == 'left':
                    new_grid.values[bottom_row][e_start:e_start+3] = [2, 2, 0]
                else:
                    new_grid.values[bottom_row][e_start:e_start+3] = [0, 2, 2]
            else:
                new_grid.values[bottom_row][e_start:e_start+3] = [2, 2, 2]

    return new_grid
