from rob_agi.colored_grid import ColoredGrid

def solve_3c9b0459(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by applying the following transformations:
    1. Rotate the grid 180 degrees (equivalent to flipping both vertically and horizontally)
    2. In the first and last rows:
       a. Move all 9s to the left side of the row
       b. Sort the remaining numbers in descending order
    
    This solution works for grids of any size and preserves the original color values
    while applying the specific transformations required by the puzzle.
    """
    # Step 1: Rotate the grid 180 degrees
    rotated = input_grid.flip_vertical().flip_horizontal()
    
    def process_row(row):
        nines = [x for x in row if x == 9]
        non_nines = sorted([x for x in row if x != 9], reverse=True)
        return nines + non_nines
    
    # Step 2: Process first and last rows
    rotated.values[0] = process_row(rotated.values[0])
    rotated.values[-1] = process_row(rotated.values[-1])
    
    return ColoredGrid(values=rotated.values)
