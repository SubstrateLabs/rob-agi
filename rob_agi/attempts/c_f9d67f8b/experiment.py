from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_f9d67f8b.main import solve_f9d67f8b

def create_test_grid():
    return ColoredGrid(values=[
        [1, 2, 3, 4, 5],
        [6, 9, 9, 9, 7],
        [8, 9, 8, 9, 8],
        [6, 9, 9, 9, 7],
        [1, 2, 3, 4, 5]
    ])

def print_grid(grid):
    for row in grid.values:
        print(" ".join(map(str, row)))

input_grid = create_test_grid()
print("Input Grid:")
print_grid(input_grid)

output_grid = solve_f9d67f8b(input_grid)
print("\nOutput Grid:")
print_grid(output_grid)

# Count the number of each color in the output grid
color_counts = {}
for row in output_grid.values:
    for cell in row:
        color_counts[cell] = color_counts.get(cell, 0) + 1

print("\nColor counts in output grid:")
for color, count in sorted(color_counts.items()):
    print(f"Color {color}: {count}")

# Check if any non-brown cells were changed
changed_non_brown = []
for i in range(5):
    for j in range(5):
        if input_grid.get_cell(i, j) != 9 and input_grid.get_cell(i, j) != output_grid.get_cell(i, j):
            changed_non_brown.append((i, j))

if changed_non_brown:
    print("\nWarning: The following non-brown cells were changed:")
    for cell in changed_non_brown:
        print(f"Row {cell[0]}, Column {cell[1]}")
else:
    print("\nAll non-brown cells remained unchanged.")
