from aider.coders import Coder
from aider.models import Model

# This is a list of files to add to the chat
fnames = ["greeting.py"]

model = Model("claude-3-5-sonnet-20240620")

# Create a coder object
coder: Coder = Coder.create(main_model=model, fnames=fnames, auto_commits=False)

# This will execute one instruction on those files and then return
coder.run("make a script that prints hello world")

# Send another instruction
coder.run("make it say goodbye")

# You can run in-chat "/" commands too
coder.run("/tokens")
