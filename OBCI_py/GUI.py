from tkinter import *
from tkinter import ttk
import numpy as np
from tkinter import messagebox
from pathlib import Path
import NIKclass
import tkinter.filedialog

months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Aout', 'Septembre', 'Octobre',
         'Novembre', 'Decembre']


class MainApplication(Tk):

    def __init__(self, names, *args, **kwargs):

        Tk.__init__(self, *args, **kwargs)
        Tk.wm_title(self, 'EEG_Analyser')
        container = Frame(self)
        container.pack()
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        self.patient_names = names
        frame = MainWindow(container, self)

        self.frames[MainWindow] = frame

        frame.grid(row=0, column=0)

        self.show_frame(MainWindow)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class MainWindow(Frame):
    nb_pieces = 0
    lengthw = 0

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label_0 = Label(self)
        label_0.pack(padx=900, pady=900)

        patient_name_list = []
        experience_nb_list = []
        for name in controller.patient_names:
            if name[:-1] not in patient_name_list:
                patient_name_list.append(name[:-1])
            if name[-1] not in experience_nb_list:
                experience_nb_list.append(name[-1])

        # ---Patient section--- #
        patient_y = 13
        patient_x = 5
        #Comboboxes
        self.patient_name_box = ttk.Combobox(self, values=patient_name_list)
        self.patient_name_box.place(y=patient_y, x=patient_x)

        self.experience_number_box = ttk.Combobox(self, values=experience_nb_list)
        self.experience_number_box.place(y=patient_y, x=patient_x+150)

        #Entry
        self.patient_list = Text(self, width=10, height=10)
        self.patient_list.place(y=patient_y, x=patient_x+350)

        #Buttons
        button_add = Button(self, text='Add', command=self.addPatient)
        button_add.place(y=patient_y+30, x=patient_x)


    def addPatient(self):
        name = self.patient_name_box.get()
        nb = self.experience_number_box.get()
        patient = name+nb
        self.patient_list.insert(END, patient + '\n')

