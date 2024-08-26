from rob_agi.colored_grid import ColoredGrid

def solve_42a15761(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a grid of 'E' shapes by fixing inconsistencies in the middle and bottom bars.
    The transformation follows these rules:
    1. Top bars and vertical segments of 'E's are always full.
    2. Middle and bottom bars alternate between full and partial (missing the rightmost square).
    3. The pattern alternates both vertically within columns and horizontally across columns.
    4. The alternation is based on the sum of the E's column index and vertical index.
    5. If this sum is even, the middle bar is full and the bottom bar is partial.
    6. If this sum is odd, the middle bar is partial and the bottom bar is full.
    7. Black vertical separating lines remain unchanged.
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
        for e in range(rows // e_height):
            e_start = col * 4
            e_top = e * e_height
            
            # Fix top bar and vertical segments
            for r in range(e_top, e_top + e_height):
                new_grid.values[r][e_start:e_start+3] = [2, 2, 2] if r == e_top else [2, 0, 2]
            
            # Determine middle and bottom bar pattern
            position_sum = col + e
            is_even = position_sum % 2 == 0
            
            # Fix middle bar
            middle_row = e_top + e_height // 2
            new_grid.values[middle_row][e_start:e_start+3] = [2, 2, 2] if is_even else [2, 2, 0]
            
            # Fix bottom bar
            bottom_row = e_top + e_height - 1
            new_grid.values[bottom_row][e_start:e_start+3] = [2, 2, 0] if is_even else [2, 2, 2]

    return new_grid
