Rough plan:

- Go through all the test and eval set:
  - Figure out roughly what it's about, what kinds of operations it needs
    - First impression
    - Visual impression
    - In words, how to solve it
    - What kind of operations are needed
  - Distill all that learning into a single set of categorizations
  - Generate an image of the problem
- Find out a bunch of primitive functions like rotate, mirror, etc. according to distribution
- Implement those functions
- If you solve something, this is important
  - We need to save a clean version of the solution
    - Description of the problem
    - Things considered
    - Short summary of the plan
    - Things that were important for solving the problem (ideas, observations, functions)
    - Solution function
- When we are in the process of solving a problem, we save our work in progress
  - Our thinking process
  - Our plan
  - Our attempts and failures, and observations
  - What we think is important for solving this problem
  - We need to let intermediate functions be executed to get information

Other notes:
When we retrieve samples from the vec db, let's do top-p sampling to not get stuck. 
Feel free to condense and re-frame the problem via LLM language
For any given problem, we will accrue information about attempts and what doesn't work
  - We need to maintain a running list of what we've learned so far
    - Important things that don't work
    - Things that we think are important for solving it
    - Summary of our attempts
We can manually intervene when we know that it is struggling on something. We need a way for human to help out, and then for that help to be digested, incorporated, then remembered.
There is maybe some basic visual processing we do
MoA to solve


Stores:
- Attempts
  - summary 
  - try number
- Research Event Chain
  - id is probably just timestamp 
  - has a previous id
  - keeps track of our current understanding of the entire challenge set
- Latest Knowledge for each problem
  - Solution (if solved)
  - Summary of our latest guess
  - Things we know about the problem
  - Things we think are important
  - Things we've tried
  - Things that don't work
  - Things that do work


====================
RESEARCH_PHASE
====================
For each problem, present what we know about it so far
  - LatestKnowledge.summary


1722754532692090000 - latest doc research
```json
{
  "current_total_knowledge": "The tasks involve transforming 2D colored grids represented as lists of lists, with colors encoded as numbers 0-9. Grids range from 1x1 to 30x30 in size. Transformations require pattern recognition, spatial reasoning, and rule application. Key aspects include: 1. Identifying color patterns, shapes, and structures within grids 2. Applying color-based transformations and mappings 3. Performing spatial operations like expanding, shifting, or compressing patterns 4. Implementing rule-based modifications based on grid structure and element relationships 5. Handling grid boundaries and edge cases 6. Executing conditional and multi-step transformations 7. Recognizing and utilizing anchor points or special colors 8. Filling shapes and propagating patterns directionally or symmetrically 9. Extracting sub-patterns and resizing grids 10. Creating frames, borders, and symmetrical arrangements Challenges often combine multiple concepts and require flexible thinking to identify and apply transformation rules to different grid sections or sub-patterns.",
  "ordered_concept_list": [
    "Pattern Recognition",
    "Color Mapping and Replacement",
    "Spatial Reasoning",
    "Rule Identification and Application",
    "Grid Structure Analysis",
    "Conditional Transformations",
    "Multi-step Operations",
    "Shape Identification and Filling",
    "Boundary and Anchor Detection",
    "Pattern Propagation",
    "Directional Transformations",
    "Symmetry and Mirroring",
    "Sub-grid Identification and Transformation",
    "Grid Resizing and Extraction",
    "Complex Shape Recognition",
    "Color-based Conditional Logic",
    "Element Preservation",
    "Frame and Border Creation",
    "Diagonal Pattern Handling",
    "Modular Transformations"
  ],
  "new_knowledge": "Recent challenges emphasize: 1. Identifying and filling hollow shapes, often based on 'seed' colors 2. Directional and symmetrical pattern propagation 3. Extracting and isolating specific patterns, resulting in grid compression 4. Complex multi-step transformations combining shape identification, filling, and propagation 5. Position-dependent transformation rules 6. Creating diagonal patterns and alternating color sequences 7. Grid expansion through pattern replication 8. Preserving certain elements while transforming others These observations highlight the need for a flexible approach capable of handling complex shapes, multiple transformation rules, and both expansion and compression of grid elements within a single challenge."
}
```
