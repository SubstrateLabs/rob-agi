from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .copy_local_file("/Users/robcheung/.ssh/id_rsa", "/root/.ssh/id_rsa")
    .run_commands("chmod 600 /root/.ssh/id_rsa", "ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts")
    .run_commands("git clone git@github.com:SubstrateLabs/rob-agi.git /app/rob_agi")
    .run_commands('pip install -e "/app/rob_agi"')
    .run_commands("git config --global user.email 'kousun12@gmail.com'", "git config --global user.name 'solver-rob'")
    .env({"GIT_COMMITTER_NAME": "solver-rob"})
)
app = modal.App("arc_solver", image=image)


@app.function(
    secrets=[modal.Secret.from_name("llm-keys")],
)
def foo():
    import rob_agi
    from rob_agi.arc_util import load_task_set
    from rob_agi.progressive_solve import Solver
    import git

    # challenge_id = "c59eb873"  # easy
    challenge_id = "776ffc46"  # hard
    task_set = "training"
    challenges, solutions = load_task_set(task_set_name=task_set)
    c = challenges[challenge_id]
    sln = solutions.get(challenge_id)
    solver = Solver(c, sln)
    solver.run_solve(max_tries=1)
    # git tag and push:

    return "hello"


@app.local_entrypoint()
def localfn():
    h = foo.remote()
    print("res", h)
