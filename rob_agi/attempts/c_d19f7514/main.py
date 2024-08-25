from rob_agi.colored_grid import ColoredGrid

def solve_d19f7514(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following rules:
    1. The output grid's height is half of the input grid's height.
    2. Process only the top half of the input grid.
    3. Change green squares (value 3) to yellow (value 4).
    4. Change black squares (value 0) to yellow (value 4) if there's a gray square (value 5) below it.
    5. Leave other colors unchanged.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed output grid.
    """
    input_values = input_grid.values
    input_height = len(input_values)
    input_width = len(input_values[0])
    output_height = input_height // 2
    
    output_values = []
    for row in range(output_height):
        new_row = []
        for col in range(input_width):
            top_cell = input_values[row][col]
            bottom_cell = input_values[row + output_height][col]
            
            if top_cell == 3:  # Green
                new_row.append(4)  # Change to Yellow
            elif top_cell == 0 and bottom_cell == 5:  # Black with Gray below
                new_row.append(4)  # Change to Yellow
            else:
                new_row.append(top_cell)  # Keep unchanged
        
        output_values.append(new_row)
    
    return ColoredGrid(values=output_values)
