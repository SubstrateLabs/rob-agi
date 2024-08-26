from rob_agi.colored_grid import ColoredGrid

def solve_bcb3040b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the bcb3040b challenge by identifying the line (row, column, or diagonal)
    with the most non-zero values and modifying it to create an alternating pattern
    of red (2) and green (3) squares. If there's a tie, it prioritizes the line with
    the most blue (1) squares, and then in the order: row, column, diagonal.
    The rest of the grid remains unchanged.
    """
    # Step 1: Analyze the grid
    rows, cols = input_grid.get_dimensions()
    counts = {
        'rows': [sum(1 for cell in row if cell != 0) for row in input_grid.values],
        'cols': [sum(1 for row in input_grid.values if row[i] != 0) for i in range(cols)],
        'diag1': sum(1 for i in range(min(rows, cols)) if input_grid.values[i][i] != 0),
        'diag2': sum(1 for i in range(min(rows, cols)) if input_grid.values[i][cols-1-i] != 0)
    }

    # Step 2: Identify the line to modify
    max_count = max(max(counts['rows']), max(counts['cols']), counts['diag1'], counts['diag2'])
    candidates = []
    
    for i, count in enumerate(counts['rows']):
        if count == max_count:
            candidates.append(('row', i))
    for i, count in enumerate(counts['cols']):
        if count == max_count:
            candidates.append(('col', i))
    if counts['diag1'] == max_count:
        candidates.append(('diag1', 0))
    if counts['diag2'] == max_count:
        candidates.append(('diag2', 0))

    # If there's a tie, choose based on number of '1' values
    if len(candidates) > 1:
        blue_counts = []
        for line_type, index in candidates:
            if line_type == 'row':
                blue_count = input_grid.values[index].count(1)
            elif line_type == 'col':
                blue_count = sum(1 for row in input_grid.values if row[index] == 1)
            elif line_type == 'diag1':
                blue_count = sum(1 for i in range(min(rows, cols)) if input_grid.values[i][i] == 1)
            else:  # diag2
                blue_count = sum(1 for i in range(min(rows, cols)) if input_grid.values[i][cols-1-i] == 1)
            blue_counts.append((blue_count, line_type, index))
        
        line_to_modify = max(blue_counts, key=lambda x: (x[0], {'row': 2, 'col': 1, 'diag1': 0, 'diag2': 0}[x[1]]))
        line_type, index = line_to_modify[1], line_to_modify[2]
    else:
        line_type, index = candidates[0]

    # Step 3: Create a new grid
    new_grid = input_grid.deep_copy()

    # Step 4: Modify the identified line
    color = 2
    if line_type == 'row':
        for col in range(cols):
            new_grid.values[index][col] = color
            color = 5 - color  # Toggle between 2 and 3
    elif line_type == 'col':
        for row in range(rows):
            new_grid.values[row][index] = color
            color = 5 - color
    elif line_type == 'diag1':
        for i in range(min(rows, cols)):
            new_grid.values[i][i] = color
            color = 5 - color
    else:  # diag2
        for i in range(min(rows, cols)):
            new_grid.values[i][cols-1-i] = color
            color = 5 - color

    # Step 5: Return the modified grid
    return new_grid
