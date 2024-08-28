from rob_agi.colored_grid import ColoredGrid

def analyze_example_4():
    input_grid = ColoredGrid(values=[
        [0, 7, 7],
        [0, 0, 0],
        [7, 7, 0]
    ])
    
    output_grid = ColoredGrid(values=[
        [0, 7, 7, 7, 0, 0],
        [0, 0, 0, 7, 0, 7],
        [7, 7, 0, 0, 0, 7],
        [0, 7, 7, 7, 0, 0],
        [0, 0, 0, 7, 0, 7],
        [7, 7, 0, 0, 0, 7]
    ])

    print("Input Grid:")
    print(input_grid)
    print("\nOutput Grid:")
    print(output_grid)

    print("\nAnalysis:")
    
    # Analyze frame creation
    print("Frame Analysis:")
    for i in range(6):
        print(f"Top: {output_grid.values[0][i]}, Bottom: {output_grid.values[5][i]}, "
              f"Left: {output_grid.values[i][0]}, Right: {output_grid.values[i][5]}")

    # Analyze interior filling
    print("\nInterior Analysis:")
    for r in range(1, 5):
        for c in range(1, 5):
            print(f"({r}, {c}): {output_grid.values[r][c]}")

    # Analyze symmetry
    print("\nSymmetry Analysis:")
    for r in range(3):
        for c in range(3):
            top_left = output_grid.values[r][c]
            top_right = output_grid.values[r][5-c]
            bottom_left = output_grid.values[5-r][c]
            bottom_right = output_grid.values[5-r][5-c]
            print(f"({r}, {c}): TL={top_left}, TR={top_right}, BL={bottom_left}, BR={bottom_right}")

    # Analyze relationship between input and output
    print("\nInput-Output Relationship:")
    for r in range(3):
        for c in range(3):
            input_val = input_grid.values[r][c]
            output_val = output_grid.values[r][c]
            print(f"Input ({r}, {c}): {input_val} -> Output: {output_val}")

if __name__ == "__main__":
    analyze_example_4()
