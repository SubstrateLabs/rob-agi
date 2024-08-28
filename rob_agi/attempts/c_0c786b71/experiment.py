from rob_agi.colored_grid import ColoredGrid

def print_grid(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print()

def analyze_transformation(input_grid, output_grid):
    print("Input Grid:")
    print_grid(input_grid.values)
    
    print("Output Grid:")
    print_grid(output_grid.values)
    
    print("Top-left 3x4 quadrant of Output Grid:")
    for row in output_grid.values[:3]:
        print(' '.join(str(cell) for cell in row[:4]))
    print()
    
    print("Transformation analysis:")
    for i in range(3):
        input_row = input_grid.values[i]
        output_row = output_grid.values[2-i][:4]  # Reverse the order and take first 4 elements
        print(f"Input row {i}: {input_row}")
        print(f"Output row {2-i}: {output_row}")
        print(f"Transformation: {' -> '.join([str(input_row), str(output_row)])}")
        print()

# Example 0
input_0 = ColoredGrid(values=[[6, 2, 4, 2], [2, 2, 6, 6], [6, 4, 2, 4]])
output_0 = ColoredGrid(values=[
    [4, 2, 4, 6, 6, 4, 2, 4],
    [6, 6, 2, 2, 2, 2, 6, 6],
    [2, 4, 2, 6, 6, 2, 4, 2],
    [2, 4, 2, 6, 6, 2, 4, 2],
    [6, 6, 2, 2, 2, 2, 6, 6],
    [4, 2, 4, 6, 6, 4, 2, 4]
])

print("Analysis for Example 0:")
analyze_transformation(input_0, output_0)

# Example 1
input_1 = ColoredGrid(values=[[5, 5, 9, 9], [9, 5, 5, 5], [5, 7, 5, 7]])
output_1 = ColoredGrid(values=[
    [7, 5, 7, 5, 5, 7, 5, 7],
    [5, 5, 5, 9, 9, 5, 5, 5],
    [9, 9, 5, 5, 5, 5, 9, 9],
    [9, 9, 5, 5, 5, 5, 9, 9],
    [5, 5, 5, 9, 9, 5, 5, 5],
    [7, 5, 7, 5, 5, 7, 5, 7]
])

print("\nAnalysis for Example 1:")
analyze_transformation(input_1, output_1)
