from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e633a9e5.main import solve_e633a9e5

def print_grid(grid):
    for row in grid.values:
        print(' '.join(map(str, row)))
    print()

# Test input
input_grid = ColoredGrid(values=[
    [6, 5, 5],
    [5, 1, 7],
    [4, 5, 2]
])

print("Input Grid:")
print_grid(input_grid)

output_grid = solve_e633a9e5(input_grid)

print("Output Grid:")
print_grid(output_grid)

# Verify 2x2 expansion
print("Verifying 2x2 expansion:")
for i in range(3):
    for j in range(3):
        input_value = input_grid.values[i][j]
        output_block = [
            output_grid.values[2*i][2*j],
            output_grid.values[2*i][2*j+1],
            output_grid.values[2*i+1][2*j],
            output_grid.values[2*i+1][2*j+1]
        ]
        print(f"Input [{i}][{j}] = {input_value}, Output 2x2 block: {output_block}")
