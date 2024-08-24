from rob_agi.colored_grid import ColoredGrid

def solve_22168020(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 22168020 challenge by expanding colors horizontally within their own "territory" in each row.
    
    The function processes each row independently:
    1. Identifies color ranges in the row
    2. Expands colors to fill their range, but only if the color appears more than once in the row
    3. Keeps single color occurrences unchanged
    4. Preserves black (0) as empty space
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with expanded color regions
    """
    def expand_row(row):
        color_ranges = {}
        for i, color in enumerate(row):
            if color != 0:
                if color not in color_ranges:
                    color_ranges[color] = [i, i]
                else:
                    color_ranges[color][1] = i
        
        new_row = row.copy()
        for color, (start, end) in color_ranges.items():
            if start != end:  # Only expand if color appears more than once
                new_row[start:end+1] = [color] * (end - start + 1)
        return new_row

    output = input_grid.deep_copy()
    for i, row in enumerate(input_grid.values):
        output.values[i] = expand_row(row)
    
    return output
