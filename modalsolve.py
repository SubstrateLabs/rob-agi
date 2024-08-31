import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("pytest", "pydantic==2.8.2", "substrate", "numpy", "aider-chat==0.54.0")
    # .pip_install("git+https://github.com/SubstrateLabs/rob-agi.git@main")
    .run_commands("git clone https://github.com/SubstrateLabs/rob-agi.git /app/rob_agi")
    .run_commands('pip install -e "/app/rob_agi"')
)
app = modal.App("arc_solver", image=image)


@app.function(
    mounts=[
        # modal.Mount.from_local_dir("data/", remote_path="/app/data"),
        # modal.Mount.from_local_dir("rob_agi/", remote_path="/app/rob_agi"),
    ],
    secrets=[modal.Secret.from_name("llm-keys")],
)
def foo():
    from rob_agi.arc_util import load_task_set
    from rob_agi.progressive_solve import Solver

    # challenge_id = "c59eb873"  # easy
    challenge_id = "776ffc46"  # hard
    task_set = "training"
    challenges, solutions = load_task_set(task_set_name=task_set)
    c = challenges[challenge_id]
    print(c)
    sln = solutions.get(challenge_id)
    solver = Solver(c, sln)

    print("\n\n" + c.test_cases[0].human_print() + "\n\n")
    return "hello"


@app.local_entrypoint()
def localfn():
    h = foo.remote()
    print("res", h)
