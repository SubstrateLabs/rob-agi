from rob_agi.colored_grid import ColoredGrid

def count_twos_in_row(grid, row_index):
    return grid.values[row_index].count(2)

def count_twos_in_column(grid, col_index):
    return sum(1 for row in grid.values if row[col_index] == 2)

def count_twos_in_main_diagonal(grid):
    rows, cols = grid.get_dimensions()
    return sum(1 for i in range(min(rows, cols)) if grid.values[i][i] == 2)

def count_twos_in_other_diagonal(grid):
    rows, cols = grid.get_dimensions()
    return sum(1 for i in range(min(rows, cols)) if grid.values[i][cols-1-i] == 2)

def find_line_with_most_twos(grid):
    rows, cols = grid.get_dimensions()
    counts = {
        'main_diagonal': count_twos_in_main_diagonal(grid),
        'other_diagonal': count_twos_in_other_diagonal(grid),
        'rows': [count_twos_in_row(grid, i) for i in range(rows)],
        'columns': [count_twos_in_column(grid, i) for i in range(cols)]
    }
    
    max_count = max(counts['main_diagonal'], counts['other_diagonal'], max(counts['rows']), max(counts['columns']))
    
    if counts['main_diagonal'] == max_count:
        return 'main_diagonal', None
    elif counts['other_diagonal'] == max_count:
        return 'other_diagonal', None
    elif max_count in counts['rows']:
        return 'row', counts['rows'].index(max_count)
    else:
        return 'column', counts['columns'].index(max_count)

def modify_diagonal(grid, is_main_diagonal):
    rows, cols = grid.get_dimensions()
    color = 2
    for i in range(min(rows, cols)):
        if is_main_diagonal:
            grid.values[i][i] = color
        else:
            grid.values[i][cols-1-i] = color
        color = 5 - color  # Toggle between 2 and 3

def modify_row(grid, row_index):
    color = 2
    for col in range(len(grid.values[row_index])):
        grid.values[row_index][col] = color
        color = 5 - color  # Toggle between 2 and 3

def modify_column(grid, col_index):
    color = 2
    for row in range(len(grid.values)):
        grid.values[row][col_index] = color
        color = 5 - color  # Toggle between 2 and 3

def solve_bcb3040b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the bcb3040b challenge by identifying the line (row, column, or diagonal)
    with the most red (2) values and modifying it to create an alternating pattern
    of red (2) and green (3) squares. If there's a tie, it prioritizes in the order:
    main diagonal, other diagonal, topmost row, leftmost column.
    The rest of the grid remains unchanged.
    """
    line_type, index = find_line_with_most_twos(input_grid)
    new_grid = input_grid.deep_copy()
    
    if line_type == 'main_diagonal':
        modify_diagonal(new_grid, True)
    elif line_type == 'other_diagonal':
        modify_diagonal(new_grid, False)
    elif line_type == 'row':
        modify_row(new_grid, index)
    else:  # column
        modify_column(new_grid, index)
    
    return new_grid
