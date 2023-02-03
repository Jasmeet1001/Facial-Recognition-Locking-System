import tkinter as tk
import recognition as rec
import process_detect as pd

from tkinter import ttk
from os import path
from threading import Thread

root=tk.Tk()
root.geometry('400x250')
userString=tk.StringVar(root)
passString=tk.StringVar(root)

def master():
    for i in root.winfo_children():
        i.destroy()

    label_1=tk.Label(root,text="Username").place(x = 60, y =50)
    inpUser=tk.Entry(root,width=30,textvariable=userString).place(x = 130, y = 50)

    label_2=tk.Label(root,text="Password").place(x = 60, y = 90)
    inpPass=tk.Entry(root,width=30,textvariable=passString,show="*").place(x = 130, y = 90)

    Button=tk.Button(root,text="Create",command=addUser).place(x = 170, y = 120)


def Login():
    for i in root.winfo_children():
        i.destroy()

    label_1=tk.Label(root,text="Username").place(x = 60, y =50)
    inpUser=tk.Entry(root,width=30,textvariable=userString).place(x = 130, y = 50)

    label_2=tk.Label(root,text="Password").place(x = 60, y = 90)
    inpPass=tk.Entry(root,width=30,textvariable=passString,show="*").place(x = 130, y = 90)

    Button=tk.Button(root,text="Submit",command=getValues).place(x = 170, y = 120)

def addUser():
    userVal=userString.get()
    passVal=passString.get()

    if path.exists(f'{userVal}'):
       label_1=tk.Label(root,text="Username already taken").place(x = 150, y =10)
    else:
        Login()

def getValues():
    userVal=userString.get()
    passVal=passString.get()

    if(userVal.rstrip()=='Jasmeet' and passVal.rstrip()=='testing123'):
    # if(userVal.rstrip()=='' and passVal.rstrip()==''):
        print("Login Sucess")        
        for i in root.winfo_children():
            i.destroy()
        main_menu()
        return userVal
    else:
        print("Failed")

clicked = tk.StringVar()

def getvall():
    return str(clicked.get()).split(' ')[0]

def threading():
    t1=Thread(target=minimize)
    t1.start()

var = tk.IntVar()
var.initialize(1)

def minimize():
    root.state(newstate='iconic')

    interval = getvall()
    username = getValues()
    inter_appli = var.get()

    if not rec.os.path.exists(f'{username}/{username}-val.pkl'):
        rec.get_name(username)
        rec.capture()
        rec.training()

    with open(f'{username}/{username}-val.pkl', 'rb') as f:
        known_faces = rec.pickle.load(f)
        known_names = rec.pickle.load(f)
        rec.get_name(username)
        if inter_appli == 1:
            rec.compare(known_faces, known_names, interval=int(interval) - 8, lock_system=pd.lock_sys)
        elif inter_appli == 2:
            pd.start_detect(known_faces, known_names)
    

def interval_func():    
    options = ttk.Combobox(root, textvariable = clicked)
    options['values'] = tuple(f'{i} secs' for i in range(10, 60+1, 5))
    options['state'] = 'readonly'
    options.place(x = 20, y = 89)

    button_inter=tk.Button(root,text="Select",command=getvall, width=10).place(x = 20, y = 115)


def selection():
    j = 1
    for i in root.winfo_children():
        if j == 7 or j == 8:
            i.destroy()
        j += 1
    
    return var.get()

def main_menu():    
    # clicked.set('5 mins')
    label=tk.Label(root,text="Detection Method", anchor=tk.W).place(x = 20, y = 20)
    
    radio_button1=tk.Radiobutton(root,text="Interval",variable=var, value=1, command=interval_func).place(x = 20, y = 40)
    radio_button2=tk.Radiobutton(root,text="Application status",variable=var, value=2, command=selection).place(x = 20, y = 60)
    
    Button_create=tk.Button(root,text="Add User",command=master, width=10).place(x = 20, y = 160)

    # disable=tk.Label(root,text="Start", anchor=tk.W).place(x = 20, y = 200)
    start_button=tk.Button(root, text='Start', command=threading, width=10).place(x = 20, y = 200)

Login()
root.mainloop()