import time
from tkinter import *
from tkinter.ttk import *
from time import strftime

root = Tk() 
root.title('Clock') 

def time(): 
	string = strftime('%H:%M:%S %p') 
	lbl.config(text = string) 
	lbl.after(1000, time) 

lbl = Label(root, font = ('calibri', 40, 'bold', 'italic'), 
			background = 'Black', 
			foreground = 'Yellow') 
lbl.pack(anchor = 'center') 
time() 
mainloop()

label = Label(root, font=("Arial", 30, 'bold'), bg="black", fg="white", bd =30)
label.grid(row =0, column=1)

def dig_clock():
    text_input = time.strftime("%H : %M : %S") 
    label.config(text=text_input)
    label.after(200, dig_clock)
    
dig_clock()
root.mainloop()