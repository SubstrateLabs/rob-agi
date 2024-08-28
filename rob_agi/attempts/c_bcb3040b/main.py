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

def modify_line(grid, line_type, index=None):
    rows, cols = grid.get_dimensions()
    original_twos = []
    line = []

    if line_type == 'main_diagonal':
        line = [grid.values[i][i] for i in range(min(rows, cols))]
    elif line_type == 'other_diagonal':
        line = [grid.values[i][cols-1-i] for i in range(min(rows, cols))]
    elif line_type == 'row':
        line = grid.values[index]
    else:  # column
        line = [row[index] for row in grid.values]

    original_twos = [i for i, val in enumerate(line) if val == 2]

    new_line = line.copy()
    for i in range(len(line)):
        if i in original_twos:
            new_line[i] = 2
        elif i > 0 and new_line[i-1] == 2:
            new_line[i] = 3
        elif i < len(line) - 1 and line[i+1] == 2:
            new_line[i] = 3
        else:
            new_line[i] = 2 if i % 2 == 1 else 3

    if line_type == 'main_diagonal':
        for i, val in enumerate(new_line):
            grid.values[i][i] = val
    elif line_type == 'other_diagonal':
        for i, val in enumerate(new_line):
            grid.values[i][cols-1-i] = val
    elif line_type == 'row':
        grid.values[index] = new_line
    else:  # column
        for i, val in enumerate(new_line):
            grid.values[i][index] = val

def solve_bcb3040b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the bcb3040b challenge by identifying the line (row, column, or diagonal)
    with the most red (2) values and modifying it to create an alternating pattern
    of red (2) and green (3) squares. The original red squares are preserved, and
    the alternating pattern is adjusted to ensure green squares are adjacent to the
    original red squares. If there's a tie for the line with the most red squares,
    it prioritizes in the order: main diagonal, other diagonal, topmost row, leftmost column.
    The rest of the grid remains unchanged.
    """
    line_type, index = find_line_with_most_twos(input_grid)
    new_grid = input_grid.deep_copy()
    modify_line(new_grid, line_type, index)
    return new_grid
