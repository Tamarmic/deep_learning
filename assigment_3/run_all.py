import os
import subprocess

modes = ["a", "b", "c", "d"]
tasks = ["pos", "ner"]

for mode in modes:
    for task in tasks:

        train_file = f"{task}/train"
        dev_file = f"{task}/dev"
        model_file = f"bilstm_{mode}_{task}.pt"
        output_file = f"{task}_{mode}.out"

        cmd = (
            f"python -u bilstmTrain.py {mode} {train_file} {model_file} "
            f"--dev_file {dev_file} --task {task}"
        )

        print(f"\nRunning: {cmd}\nLogging to: {output_file}")
        with open(output_file, "w") as out, open(output_file, "a") as err:
            process = subprocess.Popen(cmd.split(), stdout=out, stderr=err)
            process.wait()  # Wait for current job to finish before moving on
