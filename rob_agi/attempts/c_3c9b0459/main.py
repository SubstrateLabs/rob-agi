from rob_agi.colored_grid import ColoredGrid

def solve_3c9b0459(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by applying the following transformations:
    1. Rotate the grid 180 degrees (equivalent to flipping both vertically and horizontally)
    2. Swap the positions of the largest and smallest non-9 values in the first and last rows
    
    This solution works for grids of any size and preserves the original color values
    while applying the specific transformations required by the puzzle.
    """
    # Step 1: Rotate the grid 180 degrees
    rotated = input_grid.flip_vertical().flip_horizontal()
    
    # Step 2: Swap the largest and smallest non-9 values in the first and last rows
    first_row = rotated.values[0]
    last_row = rotated.values[-1]
    
    def swap_min_max(row):
        non_nine = [x for x in row if x != 9]
        if len(non_nine) >= 2:
            min_val, max_val = min(non_nine), max(non_nine)
            min_index = row.index(min_val)
            max_index = row.index(max_val)
            row[min_index], row[max_index] = max_val, min_val
    
    swap_min_max(first_row)
    swap_min_max(last_row)
    
    return ColoredGrid(values=rotated.values)
