import ctypes
import subprocess
import recognition as rec

def lock_sys():
    return ctypes.windll.user32.LockWorkStation()

def get_running_processes():
    cmd = 'powershell "gps | where {$_.MainWindowTitle} | select ProcessName'
    proc = subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)
    to_skip, running = 0, 0

    for i in proc.stdout:
        if (to_skip <= 2):
            to_skip += 1
            continue
        else:
            if (i.strip()):
                running += 1
    return running

def start_detect(known_faces, known_names,lock=lock_sys):
    starting_proc = 0
    while True:
        res = get_running_processes()
        if starting_proc == 0:
            starting_proc = res
        if res > starting_proc:
            starting_proc = res
            rec.compare(known_faces, known_names, lock)
        elif res < starting_proc:
            starting_proc = res

if(__name__ == '__main__'):
    start_detect()