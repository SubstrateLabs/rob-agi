from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_3c9b0459.main import solve_3c9b0459

def analyze_transformation(input_grid, output_grid):
    print(f"Input Grid:\n{input_grid}")
    print(f"Output Grid:\n{output_grid}")
    
    # Check if the grid is rotated 180 degrees
    rotated = input_grid.flip_vertical().flip_horizontal()
    is_rotated = rotated == output_grid
    print(f"Is grid rotated 180 degrees: {is_rotated}")
    
    # Analyze first and last rows
    for row_name, input_row, output_row in [
        ("First row", input_grid.values[0], output_grid.values[0]),
        ("Last row", input_grid.values[-1], output_grid.values[-1])
    ]:
        print(f"\n{row_name}:")
        print(f"Input: {input_row}")
        print(f"Output: {output_row}")
        
        if len(set(output_row)) == 1:
            print("All numbers are identical")
        else:
            middle = output_row[1]
            left = output_row[0]
            right = output_row[2]
            print(f"Middle: {middle}, Left: {left}, Right: {right}")
            print(f"Is middle the largest: {middle == max(output_row)}")
            print(f"Is left smaller than right: {left <= right}")
    
    print("\n" + "="*40 + "\n")

# Test cases
test_cases = [
    ColoredGrid(values=[[2, 2, 1], [2, 1, 2], [2, 8, 1]]),
    ColoredGrid(values=[[9, 2, 4], [2, 4, 4], [2, 9, 2]]),
    ColoredGrid(values=[[8, 8, 8], [5, 5, 8], [8, 5, 5]]),
    ColoredGrid(values=[[3, 2, 9], [9, 9, 9], [2, 3, 3]])
]

for input_grid in test_cases:
    output_grid = solve_3c9b0459(input_grid)
    analyze_transformation(input_grid, output_grid)
