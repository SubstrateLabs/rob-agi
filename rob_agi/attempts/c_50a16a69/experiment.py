from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_50a16a69.main import find_pattern, detect_phase_shift

def print_pattern_info(grid, name):
    pattern, pattern_height, pattern_width = find_pattern(grid)
    shift_r, shift_c = detect_phase_shift(grid, pattern)
    
    print(f"Test case: {name}")
    print(f"Input grid dimensions: {grid.get_dimensions()}")
    print(f"Detected pattern dimensions: {pattern_height}x{pattern_width}")
    print("Detected pattern:")
    for row in pattern:
        print(row)
    print(f"Detected phase shift: ({shift_r}, {shift_c})")
    print()

# Test case 1: Simple checkerboard
grid1 = ColoredGrid(values=[
    [5, 2, 5, 2],
    [2, 5, 2, 5],
    [5, 2, 5, 2],
    [2, 5, 2, 5]
])
print_pattern_info(grid1, "Simple checkerboard")

# Test case 2: Pattern with border
grid2 = ColoredGrid(values=[
    [6, 3, 5, 7, 6, 3, 5, 7, 8],
    [3, 5, 7, 6, 3, 5, 7, 6, 8],
    [6, 3, 5, 7, 6, 3, 5, 7, 8],
    [3, 5, 7, 6, 3, 5, 7, 6, 8],
    [8, 8, 8, 8, 8, 8, 8, 8, 8]
])
print_pattern_info(grid2, "Pattern with border")

# Test case 3: Pattern with phase shift
grid3 = ColoredGrid(values=[
    [7, 6, 3, 7, 6, 3],
    [3, 7, 6, 3, 7, 6],
    [6, 3, 7, 6, 3, 7],
    [7, 6, 3, 7, 6, 3],
    [3, 7, 6, 3, 7, 6],
    [6, 3, 7, 6, 3, 7]
])
print_pattern_info(grid3, "Pattern with phase shift")
