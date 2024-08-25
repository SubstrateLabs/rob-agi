from rob_agi.colored_grid import ColoredGrid

def solve_272f95fa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying sections divided by horizontal and vertical lines of 8s,
    and filling each section with specific colors:
    - Top section: Fill empty spaces with 2s
    - Bottom section: Fill empty spaces with 1s
    - Middle section:
      - Left of first vertical divider: Fill with 4s
      - Between vertical dividers: Fill with 6s
      - Right of last vertical divider: Fill with 3s
    Preserves all 8s and outer boundary 0s, including the first and last two columns.
    """
    def find_dividers(grid):
        horizontal = [i for i, row in enumerate(grid.values) if all(cell == 8 for cell in row)]
        vertical = [j for j in range(len(grid.values[0])) if all(row[j] == 8 for row in grid.values)]
        return horizontal, vertical

    def transform_section(row, col, section, left_div, right_div):
        if grid.get_cell(row, col) == 8:
            return 8
        if section == 'top':
            return 2
        if section == 'bottom':
            return 1
        if col < left_div:
            return 4
        if col > right_div:
            return 3
        return 6

    grid = input_grid.deep_copy()
    h_dividers, v_dividers = find_dividers(grid)
    
    rows, cols = grid.get_dimensions()
    
    for i in range(rows):
        for j in range(cols):
            if i < h_dividers[0]:
                section = 'top'
            elif i > h_dividers[-1]:
                section = 'bottom'
            else:
                section = 'middle'
            
            if section == 'middle':
                left_div = v_dividers[0]
                right_div = v_dividers[-1]
            else:
                left_div = right_div = -1
            
            # Preserve outer boundary 0s (including first and last two columns) and all 8s
            if (i == 0 or i == rows - 1 or j <= 1 or j >= cols - 2) and grid.get_cell(i, j) == 0:
                continue
            if grid.get_cell(i, j) == 8:
                continue
            
            new_value = transform_section(i, j, section, left_div, right_div)
            grid.set_cell(i, j, new_value)
    
    return grid
