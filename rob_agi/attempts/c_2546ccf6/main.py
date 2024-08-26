from rob_agi.colored_grid import ColoredGrid

def solve_2546ccf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating horizontal and vertical symmetry.
    
    1. Identifies main vertical dividing lines.
    2. Splits the grid into left, middle, and right sections.
    3. Mirrors the left section horizontally to the right section.
    4. Creates vertical symmetry in the middle section.
    5. Preserves all dividing lines (colors 2 or 6).
    
    Returns a new grid with the transformed pattern.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    # Find vertical dividing lines
    dividers = [i for i in range(cols) if grid.values[0][i] in [2, 6]]
    if len(dividers) < 2:
        return grid  # Not enough dividers, return original grid
    
    left_div, right_div = dividers[0], dividers[-1]
    
    # Mirror left section to right
    for r in range(rows):
        left_section = grid.values[r][:left_div]
        for c in range(right_div + 1, cols):
            if grid.values[r][c] not in [2, 6]:  # Preserve dividing lines
                grid.values[r][c] = left_section[cols - 1 - c]
    
    # Create vertical symmetry in middle section
    mid_col = (left_div + right_div) // 2
    for c in range(left_div + 1, right_div):
        top_half = [grid.values[r][c] for r in range(rows // 2)]
        for r in range(rows // 2, rows):
            if grid.values[r][c] not in [2, 6]:  # Preserve dividing lines
                grid.values[r][c] = top_half[rows - 1 - r]
    
    return grid
