from rob_agi.colored_grid import ColoredGrid

def solve_3c9b0459(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by applying the following transformations:
    1. Rotate the grid 180 degrees (equivalent to flipping both vertically and horizontally)
    2. In the first and last rows:
       a. Swap the positions of the largest and smallest non-9 values
       b. Move all 9s to the left side of the row
    
    This solution works for grids of any size and preserves the original color values
    while applying the specific transformations required by the puzzle.
    """
    # Step 1: Rotate the grid 180 degrees
    rotated = input_grid.flip_vertical().flip_horizontal()
    
    def process_row(row):
        nines = [x for x in row if x == 9]
        non_nines = [x for x in row if x != 9]
        if len(non_nines) >= 2:
            min_val, max_val = min(non_nines), max(non_nines)
            non_nines.remove(min_val)
            non_nines.remove(max_val)
            return nines + [max_val] + non_nines + [min_val]
        return row
    
    # Step 2: Process first and last rows
    rotated.values[0] = process_row(rotated.values[0])
    rotated.values[-1] = process_row(rotated.values[-1])
    
    return ColoredGrid(values=rotated.values)
