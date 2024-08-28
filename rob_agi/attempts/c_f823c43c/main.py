from rob_agi.colored_grid import ColoredGrid
from collections import Counter, defaultdict

def solve_f823c43c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the background color and the pattern color,
    then creates a new grid with a regular pattern based on these colors.
    
    1. Analyzes the input grid to find color frequencies and patterns.
    2. Determines the background color (most frequent) and pattern color (most consistent pattern).
    3. Creates a new grid filled with the background color.
    4. Applies the pattern color in a regular grid based on the identified intervals.
    5. Returns the new grid as a ColoredGrid object.
    """
    rows, cols = input_grid.get_dimensions()
    color_counts = Counter(cell for row in input_grid.values for cell in row)
    
    def analyze_pattern(color):
        occurrences = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == color]
        row_intervals = defaultdict(int)
        col_intervals = defaultdict(int)
        for i in range(len(occurrences)):
            for j in range(i+1, len(occurrences)):
                r1, c1 = occurrences[i]
                r2, c2 = occurrences[j]
                if r1 == r2:
                    row_intervals[c2 - c1] += 1
                if c1 == c2:
                    col_intervals[r2 - r1] += 1
        row_interval = max(row_intervals, key=row_intervals.get, default=1)
        col_interval = max(col_intervals, key=col_intervals.get, default=1)
        consistency = (row_intervals[row_interval] + col_intervals[col_interval]) / (len(occurrences) * (len(occurrences) - 1) / 2) if occurrences else 0
        return consistency, row_interval, col_interval, occurrences[0] if occurrences else None

    color_patterns = {color: analyze_pattern(color) for color in color_counts}
    pattern_color = max(color_patterns, key=lambda x: (color_patterns[x][0], color_counts[x]))
    background_color = max(color_counts, key=lambda x: color_counts[x] if x != pattern_color else 0)

    new_grid = [[background_color for _ in range(cols)] for _ in range(rows)]
    
    _, row_interval, col_interval, start = color_patterns[pattern_color]
    if start:
        start_row, start_col = start
        for r in range(rows):
            for c in range(cols):
                if (r - start_row) % row_interval == 0 and (c - start_col) % col_interval == 0:
                    new_grid[r][c] = pattern_color

    return ColoredGrid(values=new_grid)
