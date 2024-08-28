from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_e6de6e8f.main import solve_e6de6e8f

def print_grid(grid):
    for row in grid.values:
        print(''.join(str(cell) for cell in row))
    print()

def analyze_input_output(input_grid, output_grid):
    print("Input Grid:")
    print_grid(input_grid)
    print("Output Grid:")
    print_grid(output_grid)
    
    decision_points = [i for i, v in enumerate(input_grid.values[0]) if v == 2]
    print(f"Decision points: {decision_points}")
    
    bottom_row_reds = [i for i, v in enumerate(input_grid.values[1]) if v == 2]
    print(f"Bottom row reds: {bottom_row_reds}")
    
    path = []
    for i, row in enumerate(output_grid.values):
        if 2 in row:
            path.append((i, row.index(2)))
    print(f"Path: {path}")
    
    diagonal_moves = sum(1 for (r1, c1), (r2, c2) in zip(path, path[1:]) if r2-r1 == 1 and c2-c1 == 1)
    print(f"Number of diagonal moves: {diagonal_moves}")

# Test cases
test_cases = [
    ([[2, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2], [2, 2, 0, 2, 2, 0, 2, 0, 2, 2, 0, 2]]),
    ([[0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2], [2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 2]]),
    ([[2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2], [2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 2]])
]

for i, case in enumerate(test_cases):
    print(f"\nTest Case {i + 1}:")
    input_grid = ColoredGrid(values=case)
    output_grid = solve_e6de6e8f(input_grid)
    analyze_input_output(input_grid, output_grid)
