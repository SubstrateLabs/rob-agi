from rob_agi.colored_grid import ColoredGrid

def solve_42a15761(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a grid of 'E' shapes by fixing inconsistencies in the middle and bottom bars.
    The transformation follows these rules:
    1. Top bars and vertical segments of 'E's are always full.
    2. Middle bars alternate between full and partial (missing the rightmost square) within each column and across columns.
    3. Bottom bars are the opposite of middle bars: full when the middle bar is partial, and partial (missing the rightmost square) when the middle bar is full.
    4. The pattern of middle bars alternates between columns: odd columns start with a full middle bar, even columns start with a partial middle bar.
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

    e_height = get_e_height()
    num_columns = get_num_columns()
    
    for col in range(num_columns):
        start_full = col % 2 == 0  # Odd columns (0-indexed) start with full middle bar
        for e in range(rows // e_height):
            e_start = col * 4
            e_top = e * e_height
            
            # Fix top bar and vertical segments
            for r in range(e_top, e_top + e_height):
                new_grid.values[r][e_start:e_start+3] = [2, 2, 2] if r == e_top else [2, 0, 2]
            
            # Fix middle bar
            middle_row = e_top + e_height // 2
            is_full_middle = start_full if e % 2 == 0 else not start_full
            new_grid.values[middle_row][e_start:e_start+3] = [2, 2, 2] if is_full_middle else [2, 2, 0]
            
            # Fix bottom bar
            bottom_row = e_top + e_height - 1
            new_grid.values[bottom_row][e_start:e_start+3] = [2, 2, 0] if is_full_middle else [2, 2, 2]

    return new_grid
