from rob_agi.colored_grid import ColoredGrid

def analyze_dot_distribution(grid: ColoredGrid, base_line_row: int, dot_color: int):
    rows, cols = grid.get_dimensions()
    above_count = [0] * cols
    below_count = [0] * cols
    
    for c in range(cols):
        if grid.get_cell(base_line_row - 1, c) == dot_color:
            above_count[c] = 1
        if grid.get_cell(base_line_row + 1, c) == dot_color:
            below_count[c] = 1
    
    print(f"Dots above base line: {above_count}")
    print(f"Dots below base line: {below_count}")
    print(f"Total dots above: {sum(above_count)}")
    print(f"Total dots below: {sum(below_count)}")

# Example 0 expected output
expected = ColoredGrid(values=
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [5, 5, 5, 0, 0, 5, 5, 0, 0, 5],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [5, 5, 5, 5, 0, 0, 5, 5, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)

analyze_dot_distribution(expected, 8, 5)
