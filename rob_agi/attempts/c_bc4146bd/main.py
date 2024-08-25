from rob_agi.colored_grid import ColoredGrid

def solve_bc4146bd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 4x4 input grid into a 4x20 output grid by repeating the pattern five times.
    The pattern alternates between direct copying and mirroring of the original 4x4 grid.
    
    1. Copy the original 4x4 grid (columns 0-3)
    2. Mirror the 4x4 grid horizontally (columns 4-7)
    3. Repeat steps 1-2 until the 4x20 grid is filled
    """
    input_rows, input_cols = input_grid.get_dimensions()
    new_rows, new_cols = 4, 20
    new_values = []
    
    for row in range(new_rows):
        new_row = []
        for col in range(new_cols):
            set_of_four = col // 4
            
            if set_of_four % 2 == 0:
                input_col = col % 4
            else:
                input_col = 3 - (col % 4)
            
            color = input_grid.values[row][input_col]
            new_row.append(color)
        
        new_values.append(new_row)
    
    result = ColoredGrid(values=new_values)
    return result
