from rob_agi.colored_grid import ColoredGrid

def solve_2546ccf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating horizontal and vertical symmetry.
    
    1. Identifies main vertical dividing lines.
    2. Splits the grid into four vertical sections.
    3. Applies horizontal symmetry by mirroring outer sections to inner sections.
    4. Applies vertical symmetry within each section.
    5. Preserves all dividing lines (colors 2 or 6).
    
    Returns a new grid with the transformed pattern.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    # Find vertical dividing lines
    dividers = [i for i in range(cols) if grid.values[0][i] in [2, 6]]
    if len(dividers) < 3:
        return grid  # Not enough dividers, return original grid
    
    left_div, mid_div, right_div = dividers[0], dividers[1], dividers[2]
    
    # Apply horizontal symmetry
    for r in range(rows):
        left_section = grid.values[r][:left_div]
        right_section = grid.values[r][right_div+1:]
        for c in range(left_div + 1, mid_div):
            if grid.values[r][c] not in [2, 6]:
                grid.values[r][c] = right_section[c - (left_div + 1)]
        for c in range(mid_div + 1, right_div):
            if grid.values[r][c] not in [2, 6]:
                grid.values[r][c] = left_section[left_div - 1 - (c - mid_div)]
    
    # Apply vertical symmetry
    half_rows = rows // 2
    for section_start, section_end in [(0, left_div), (left_div + 1, mid_div), (mid_div + 1, right_div), (right_div + 1, cols)]:
        for c in range(section_start, section_end):
            for r in range(half_rows):
                if grid.values[rows - 1 - r][c] not in [2, 6]:
                    grid.values[rows - 1 - r][c] = grid.values[r][c]
    
    return grid
