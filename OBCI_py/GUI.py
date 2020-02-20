from tkinter import *
from tkinter import ttk
import numpy as np
from tkinter import messagebox
from pathlib import Path
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

        self.sequence_plan = [4, 2, 3, 5, 1, 2, 5, 4, 2, 3, 1, 5, 4, 3, 2, 4, 1, 2, 5, 3, 4, 1, 3, 1, 3]
        self.freq_possible = [12.00, 10.00, 8.57, 7.50, 6.66]
        self.frequencies_plan = []
        for s in self.sequence_plan:
            self.frequencies_plan.append(self.freq_possible[s-1])

        self.request = []
        self.ready_to_send = False

        # ---Patient section--- #
        patient_y = 13
        patient_x = 5

        # Comboboxes
        self.patient_name_box = ttk.Combobox(self, values=patient_name_list)
        self.patient_name_box.place(y=patient_y, x=patient_x)

        self.experience_number_box = ttk.Combobox(self, values=experience_nb_list)
        self.experience_number_box.place(y=patient_y, x=patient_x+170)

        # Textbox
        self.patient_list = Text(self, width=10, height=10)
        self.patient_list.place(y=patient_y, x=patient_x+350)

        # Buttons
        button_add = Button(self, text='Add', command=self.addPatient)
        button_add.place(y=patient_y+30, x=patient_x)

        button_clear = Button(self, text='Clear', command=self.clearTextbox)
        button_clear.place(y=patient_y+120, x=patient_x+350)

        # ---Settings section--- #
        y_setting = 30
        y_spacing = 20
        x_setting = 500

        # Buttons
        button_request = Button(self, text='Send', command=self.getRequest)
        button_request.place(y=y_setting+120, x=x_setting+100)

        # Checkboxes
        self.checkboxes_list = []
        self.checkbox_names = ['Fft', 'Spectrum', 'Success Rate', 'Frequencies Summary']
        for i in range(4):
            self.checkboxes_list.append(ttk.Checkbutton(self, text=' ' + self.checkbox_names[i]))
            self.checkboxes_list[i].place(y=y_setting+y_spacing*i, x=x_setting)

        # Label
        self.seq_text = StringVar()
        self.seq_text.set('Seq nb : 4 - 7.50 Hz')
        self.seq_label = Label(self, textvariable=self.seq_text)
        self.seq_label.place(y=y_setting, x=x_setting+350)

        self.sucr_text = StringVar()
        self.sucr_text.set('Succès global : ')
        self.sucr_label = Label(self, textvariable=self.sucr_text)
        self.sucr_label.place(y=y_setting, x=x_setting+250)

        # Slider
        self.channel_slider = Scale(self, from_=1, to=14, orient=HORIZONTAL, length=100)
        self.channel_slider.place(y=y_setting - 20, x=x_setting + 120)

        self.sequence_slider = Scale(self, from_=1, to=25, orient=HORIZONTAL, length=200, command=self.dispSliderVal)
        self.sequence_slider.place(y=y_setting+10, x=x_setting+120)


    def dispSliderVal(self, idx):
        idx = int(idx)
        self.seq_text.set('Seq nb : ' + str(self.sequence_plan[idx-1]) + ' - ' + str(self.frequencies_plan[idx-1]) + ' Hz')

    def clearTextbox(self):
        self.patient_list.delete(1.0, END)
        self.allowedFunc()

    def allowedFunc(self):
        if len(self.patient_list.get(1.0, END)) > 7:
            self.checkboxes_list[0].state(['!selected'])
            self.checkboxes_list[0].state(['disabled'])
        else:
            self.checkboxes_list[0].state(['!disabled'])

    def addPatient(self):
        name = self.patient_name_box.get()
        nb = self.experience_number_box.get()
        patient = name+nb
        self.patient_list.insert(END, patient + '\n')
        self.allowedFunc()

    def getPatients(self):
        p_list = self.patient_list.get("1.0", END)
        return p_list

    def getSettings(self):
        settings = []
        settings_values = []
        for c in self.checkboxes_list:
            settings.append(c.state())

        settings_values.append(self.channel_slider.get())
        settings_values.append(self.sequence_slider.get())

        return settings, settings_values

    def getRequest(self):
        # ---Nom des patients requested--- #
        patients = self.getPatients()
        p = []
        name = ''
        for letter in patients:
            if letter != '\n':
                name += letter
            else:
                if name != '' and name not in p:
                    p.append(name)
                name = ''

        # ---SucessRate ---#
        self.sucr_text.set('Succès global : {} %'.format('à implémenter'))

        # ---Settings--- #
        s, sv = self.getSettings()
        self.request = [p, s, sv]
        self.ready_to_send = True
        self.quit()

