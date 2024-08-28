from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_dd2401ed.main import solve_dd2401ed

def print_grid(grid):
    for row in grid.values:
        print(' '.join(str(cell) for cell in row))
    print()

# Create a simple test grid
test_grid = ColoredGrid(values=[
    [0, 0, 1, 5, 0, 2, 0, 0],
    [0, 1, 0, 5, 0, 0, 2, 0],
    [0, 0, 0, 5, 2, 0, 0, 0]
])

print("Original grid:")
print_grid(test_grid)

result = solve_dd2401ed(test_grid)

print("Transformed grid:")
print_grid(result)

# Check if blue dots remained in their original positions
blue_dots_original = [(r, c) for r in range(len(test_grid.values)) for c in range(len(test_grid.values[0])) if test_grid.values[r][c] == 1]
blue_dots_result = [(r, c) for r in range(len(result.values)) for c in range(len(result.values[0])) if result.values[r][c] == 1]

print("Blue dots in original positions:", blue_dots_original)
print("Blue dots in result:", blue_dots_result)

if blue_dots_original == blue_dots_result:
    print("Blue dots remained in their original positions.")
else:
    print("Blue dots changed positions or color!")

# Check red dots movement
red_dots_original = [(r, c) for r in range(len(test_grid.values)) for c in range(len(test_grid.values[0])) if test_grid.values[r][c] == 2]
red_dots_result = [(r, c) for r in range(len(result.values)) for c in range(len(result.values[0])) if result.values[r][c] == 2]

print("Red dots in original positions:", red_dots_original)
print("Red dots in result:", red_dots_result)

# Calculate the shift of the gray line
original_gray = next(c for c in range(len(test_grid.values[0])) if test_grid.values[0][c] == 5)
result_gray = next(c for c in range(len(result.values[0])) if result.values[0][c] == 5)
shift = result_gray - original_gray

print(f"Gray line shifted by {shift} columns")

# Check if red dots shifted correctly
correct_shift = all((r, (c - shift) % len(test_grid.values[0])) in red_dots_result for r, c in red_dots_original)
print("Red dots shifted correctly:", correct_shift)
