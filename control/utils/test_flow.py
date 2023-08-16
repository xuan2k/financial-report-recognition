import os
import subprocess
from tqdm import tqdm

# Run table detection
start_docker= "docker start tabdet"

subprocess.run(start_docker, shell=True, check=True)


code_to_run_in_container = f"cd source/pretrained-test && ls && python table_det.py"
subprocess.run(["docker", "exec", "tabdet", "sh", "-c", code_to_run_in_container])

stop_docker= "docker stop tabdet"

subprocess.run(stop_docker, shell=True, check=True)

# # Run table structure recognition
# start_table_tsr_docker= "docker start table_res"

# subprocess.run(start_table_tsr_docker, shell=True, check=True)

# code_to_run_in_container = "cd source/networks && python table_tsr.py"
# subprocess.run(["docker", "exec", "table_res", "sh", "-c", code_to_run_in_container])

# stop_docker= "docker stop table_res"

# subprocess.run(stop_docker, shell=True, check=True)

# # Run text detection
# code_to_run_in_container = "cd networks && python3 text_det.py"
# subprocess.run(code_to_run_in_container, shell=True, check=True)

# # Run text detection
# code_to_run_in_container = "cd networks && python3 text_rec.py"
# subprocess.run(code_to_run_in_container, shell=True, check=True)

# code_to_run_in_container = "cd networks && python3 table_processing.py"
# subprocess.run(code_to_run_in_container, shell=True, check=True)


# code_to_run_in_container = "cd test && python3 visualize.py"
# subprocess.run(code_to_run_in_container, shell=True, check=True)

working_directory = "/home/xuan/Project/OCR"

activation = "./active.sh"

subprocess.run(activation, shell=True, cwd=working_directory)


