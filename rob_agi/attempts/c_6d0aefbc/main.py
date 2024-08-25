from rob_agi.colored_grid import ColoredGrid

def solve_6d0aefbc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its width. The left half of the output
    is identical to the input, while the right half is a horizontally mirrored
    version of the left half.
    """
    # Get the dimensions of the input grid
    rows, cols = input_grid.get_dimensions()
    
    # Create a new grid with double the width
    new_grid = []
    for row in range(rows):
        original_row = [input_grid.get_cell(row, col) for col in range(cols)]
        mirrored_row = original_row[::-1]  # Reverse the original row
        new_row = original_row + mirrored_row
        new_grid.append(new_row)
    
    # Create and return a new ColoredGrid with the transformed values
    return ColoredGrid(values=new_grid)
