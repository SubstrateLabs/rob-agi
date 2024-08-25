from rob_agi.colored_grid import ColoredGrid

def solve_e21a174a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the e21a174a challenge by rearranging color groups vertically.
    
    The function identifies distinct color groups in the input grid,
    preserves their internal structure and the empty space above each group,
    and then rearranges them from bottom to top in reverse order of their
    original positions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with color groups rearranged.
    """
    rows, cols = len(input_grid.values), len(input_grid.values[0])
    
    # Step 1: Analyze the input grid
    color_groups = []
    current_group = None
    empty_rows_above = 0
    
    for row in range(rows):
        row_colors = set(input_grid.values[row]) - {0}
        if not row_colors:
            empty_rows_above += 1
            continue
        
        if current_group and current_group['color'] in row_colors:
            current_group['bottom'] = row
            current_group['cells'].extend((row, col) for col in range(cols) if input_grid.values[row][col] == current_group['color'])
        else:
            if current_group:
                color_groups.append(current_group)
            current_group = {
                'color': next(iter(row_colors)),
                'top': row,
                'bottom': row,
                'empty_above': empty_rows_above,
                'cells': [(row, col) for col in range(cols) if input_grid.values[row][col] in row_colors]
            }
            empty_rows_above = 0
    
    if current_group:
        color_groups.append(current_group)
    
    # Step 2: Sort color groups from bottom to top
    color_groups.sort(key=lambda g: g['bottom'], reverse=True)
    
    # Step 3: Calculate new positions
    current_row = rows - 1
    for group in color_groups:
        group_height = group['bottom'] - group['top'] + 1
        new_bottom = current_row
        new_top = new_bottom - group_height + 1
        group['new_top'] = new_top
        current_row = new_top - group['empty_above'] - 1
    
    # Step 4: Create the output grid
    output_values = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Step 5: Place color groups in new positions
    for group in color_groups:
        vertical_offset = group['new_top'] - group['top']
        for row, col in group['cells']:
            new_row = row + vertical_offset
            output_values[new_row][col] = group['color']
    
    # Step 6: Return the new ColoredGrid
    return ColoredGrid(values=output_values)
