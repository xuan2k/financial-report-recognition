import subprocess

# Run table detection
start_docker= "docker start tabdet"

subprocess.run(start_docker, shell=True, check=True)

code_to_run_in_container = "cd source && make table_det_demo"
subprocess.run(["docker", "exec", "tabdet", "sh", "-c", code_to_run_in_container])

stop_docker= "docker stop tabdet"

subprocess.run(stop_docker, shell=True, check=True)

# Run table structure recognition
start_table_tsr_docker= "docker start table_res"

subprocess.run(start_table_tsr_docker, shell=True, check=True)

code_to_run_in_container = "cd source && make table_tsr_demo"
subprocess.run(["docker", "exec", "table_res", "sh", "-c", code_to_run_in_container])

stop_docker= "docker stop table_res"

subprocess.run(stop_docker, shell=True, check=True)

working_directory = "/home/xuan/Project/OCR"

activation = "./active.sh"

subprocess.run(activation, shell=True, cwd=working_directory)


