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