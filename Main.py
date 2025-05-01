# Import necessary libraries and modules
import tkinter as tk
import pandas as pd
from tkinter import *
import tkinter.messagebox as tm
import numpy as np
from tkinter import filedialog
import tkinter.messagebox as tm
import easygui # For creating easy graphical interfaces for input

# Custom imports for machine learning algorithms and preprocessing
import Logisticregression as LR
import RF as RF
import DT as dt
import GB as gb
import LGBM as lgbm
import KNN as knn
import votingclassifier as vc 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define global colors for UI design
bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"


def Home():
        # Function to create the main application window
        global window

        # Function to clear the text field
        def clear():
            print("Clear1")
            txt.delete(0, 'end')    


        window = tk.Tk()
        window.title("DIAGNOSIS OF  POLYCYSTIC OVARY SYNDROME USING MACHINE LEARNING ALGORITHMS")
        
 
        window.geometry('1580x960')  # Size of the window
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="DIAGNOSIS OF  POLYCYSTIC OVARY SYNDROME USING ML" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=2,font=('times', 30, 'italic bold underline')) 
        message1.place(x=100, y=1)

        lbl = tk.Label(window, text="Dataset",width=10  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl.place(x=1, y=100)
        
        txt = tk.Entry(window,width=30,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt.place(x=200, y=110)
       

        

        # Function to browse and select a file
        def browse():
                path=filedialog.askopenfilename()
                print(path)
                txt.delete(0, 'end')
                txt.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Datset")     

        
        # Function to process data using Logistic Regression
        def LRprocess():
                # Retrieve the dataset path from the text entry widget
                sym = txt.get()
                if sym != "":
                        # Call the process function from the 'LR' module, passing the dataset path
                        LR.process(sym)
                        # Display a success message box
                        tm.showinfo("Input", "Logistic Regression Successfully Finished")
                else:
                        # Display an error message if no dataset is selected
                        tm.showinfo("Input error", "Select Dataset File")

        # Function to process data using Random Forest algorithm
        def RFprocess():
                # Retrieve the dataset path from the text entry widget
                sym = txt.get()
                if sym != "":
                        # Call the process function from the 'RF' module, passing the dataset path
                        RF.process(sym)
                        # Display a success message box
                        tm.showinfo("Input", "Random Forest Successfully Finished")
                else:
                        # Display an error message if no dataset is selected
                        tm.showinfo("Input error", "Select Dataset File")

        # Function to process data using Decision Tree algorithm
        def DTprocess():
                # Retrieve the dataset path from the text entry widget
                sym = txt.get()
                if sym != "":
                        # Call the process function from the 'dt' module, passing the dataset path
                        dt.process(sym)
                        # Display a success message box
                        tm.showinfo("Input", "Decision Tree Successfully Finished")
                else:
                        # Display an error message if no dataset is selected
                        tm.showinfo("Input error", "Select Dataset File")


        # Function to process data using Gradient Boosting
        def GBprocess():
                # Retrieve the dataset path from the text entry widget
                sym = txt.get()
                if sym != "":
                        # Call the process function from the 'gb' module, passing the dataset path
                        gb.process(sym)
                        # Display a success message box
                        tm.showinfo("Input", "Gradient Boosting Successfully Finished")
                else:
                        # Display an error message if no dataset is selected
                        tm.showinfo("Input error", "Select Dataset File")

        # Function to process data using Light Gradient Boosting Machine
        def LGBMprocess():
                # Retrieve the dataset path from the text entry widget
                sym = txt.get()
                if sym != "":
                        # Call the process function from the 'lgbm' module, passing the dataset path
                        lgbm.process(sym)
                        # Display a success message box
                        tm.showinfo("Input", "Light Gradient Boosting Successfully Finished")
                else:
                        # Display an error message if no dataset is selected
                        tm.showinfo("Input error", "Select Dataset File")
        # Function to execute a Voting Classifier
        def VCprocess():
                # Retrieve the dataset path from the text entry widget
                sym = txt.get()
                if sym != "":
                        # Call the process function from the 'vc' module, passing the dataset path
                        vc.process(sym)
                        # Display a success message box indicating the completion of processing with the Voting Classifier
                        tm.showinfo("Input", "Voting Classifier Successfully Finished")
                else:
                        # Display an error message if no dataset is selected
                        tm.showinfo("Input error", "Select Dataset File")
        
        # Function to process data using K-Nearest Neighbors
        def KNNprocess():
                # Retrieve the dataset path from the text entry widget
                sym = txt.get()
                if sym != "":
                        # Call the process function from the 'knn' module, passing the dataset path
                        knn.process(sym)
                        # Display a success message box
                        tm.showinfo("Input", "KNN Successfully Finished")
                else:
                        # Display an error message if no dataset is selected
                        tm.showinfo("Input error", "Select Dataset File")
                
        # This function collects user inputs via easygui, predicts using a trained model, and displays the result
        
        def Predictprocess():
                # Collecting user inputs for health-related features
                Age = int(easygui.enterbox("Enter Age "))
                weight = float(easygui.enterbox("Weight (Kg)"))
                height = int(easygui.enterbox("Height (Cm)"))
                BMI = float(easygui.enterbox("BMI"))
                bgroup = easygui.enterbox("Blood Group(O+ ve)")

                # Encoding blood group into numerical values for model processing
                blood_groups = {"A+ ve": 11, "A- ve": 12, "B+ ve": 13, "B- ve": 14, "O+ ve": 15, "O- ve": 16, "AB+ ve": 17, "AB- ve": 18}
                bgroup = blood_groups.get(bgroup, bgroup)  # Default to original input if not found

                # Additional user inputs
                paulerate = int(easygui.enterbox("Pulse rate(bpm)"))
                rr = int(easygui.enterbox("RR (breaths/min)"))
                hb = float(easygui.enterbox("Hb(g/dl)"))
                ct = easygui.enterbox("Cycle(Regular/Irregular)")
                ct = 2 if ct == "Regular" else 4  # Encoding cycle regularity
                # More health metrics from user input
                cylen = int(easygui.enterbox("Cycle length(days)"))
                marry = int(easygui.enterbox("Marraige Status (Yrs)"))
                prag = 1 if easygui.enterbox("Pregnant(Yes/No)") == "Yes" else 0
                noa = int(easygui.enterbox("No. of aborptions"))
                IHCG = float(easygui.enterbox("I   beta-HCG(mIU/mL)"))
                IIHCG = float(easygui.enterbox("II   beta-HCG(mIU/mL)"))
                fsh = float(easygui.enterbox("FSH(mIU/mL)"))
                lh = float(easygui.enterbox("LH(mIU/mL)"))
                fshdividebylh = float(easygui.enterbox("FSH/LH"))
                hip = float(easygui.enterbox("Hip(inch)"))
                waist = float(easygui.enterbox("Waist(inch)"))
                whr = float(easygui.enterbox("Waist:Hip Ratio"))
                tsh = float(easygui.enterbox("TSH (mIU/L)"))
                amh = float(easygui.enterbox("AMH(ng/mL)"))
                prl = float(easygui.enterbox("PRL(ng/mL)"))
                v3 = float(easygui.enterbox("Vit D3 (ng/mL)"))
                prg = float(easygui.enterbox("PRG(ng/mL)"))
                rbs = float(easygui.enterbox("RBS(mg/dl)"))
                
                # Collecting user inputs for lifestyle and symptoms
                wg = 1 if easygui.enterbox("Weight gain(Yes/No)") == "Yes" else 0
                hg = 1 if easygui.enterbox("hair growth(Yes/No)") == "Yes" else 0
                sd = 1 if easygui.enterbox("Skin darkening(Yes/No)") == "Yes" else 0
                hl = 1 if easygui.enterbox("Hair loss(Yes/No)") == "Yes" else 0
                pim = 1 if easygui.enterbox("Pimples (Yes/No)") == "Yes" else 0
                fd = 1 if easygui.enterbox("Fast food(Yes/No)") == "Yes" else 0
                re = 1 if easygui.enterbox("Reg.Exercise(Yes/No)") == "Yes" else 0
                # Blood pressure and follicle data      
                bpsys=float(easygui.enterbox("BP _Systolic (mmHg)"))
                bpdia=float(easygui.enterbox("BP _Diastolic (mmHg)"))
                fnol=float(easygui.enterbox("Follicle No. (L)"))
                fnor=float(easygui.enterbox("Follicle No. (R))"))
                afsl=float(easygui.enterbox("Avg. F size (L) (mm)"))
                afsr=float(easygui.enterbox("Avg. F size (R) (mm)"))
                endom=float(easygui.enterbox("Endometrium (mm)"))

                # Loading the dataset for prediction
                data=pd.read_csv("data.csv")    
                X=data.drop(["PCOS (Y/N)","Sl. No","Patient File No."],axis = 1) #droping out index from features too
                y=data["PCOS (Y/N)"]
                #Splitting the data into test and training sets
                X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
                #Fitting the RandomForestClassifier to the training set
                rfc = RandomForestClassifier()
                rfc.fit(X_train, y_train)
                 
                # Prepare user input for prediction
                x_text=[float(Age),float(weight),float(height),float(BMI),float(bgroup),float(paulerate),float(rr),float(hb),float(ct),float(cylen),float(marry),float(prag),float(noa),float(IHCG),float(IIHCG),float(fsh),float(lh),float(fshdividebylh),float(hip),float(waist),float(whr),float(tsh),float(amh),float(prl),float(v3),float(prg),float(rbs),float(wg),float(hg),float(sd),float(hl),float(pim),float(fd),float(re),float(bpsys),float(bpdia),float(fnol),float(fnor),float(afsl),float(afsr),float(endom)]
                print("x_text",x_text)
                inputdata=x_text
                x_text=np.array(x_text)
                x_text=x_text.reshape(1, -1)
                
                # Predicting the outcome
                res=rfc.predict(x_text)
                msg=""
                if res[0]==0:
                        msg="No disease"
                else:
                        msg="You have infected by POCOS"
                tm.showinfo("Output", "Prediction : " +str(msg))
      

       # Button to open a file dialog to browse and select a dataset file
        browse = tk.Button(window, text="Browse", command=browse, fg=fgcolor, bg=bgcolor1, width=20, height=1, activebackground="Red", font=('times', 15, 'bold'))
        browse.place(x=600, y=110)  # Positioning the button on the window

        # Button to clear any data in the input fields or selections
        clearButton = tk.Button(window, text="Clear", command=clear, fg=fgcolor, bg=bgcolor1, width=20, height=1, activebackground="Red", font=('times', 15, 'bold'))
        clearButton.place(x=900, y=110)  # Positioning the button next to the browse button

        # Button for initiating Logistic Regression processing
        LRbutton = tk.Button(window, text="LogisticRegression", command=LRprocess, fg=fgcolor, bg=bgcolor1, width=14, height=1, activebackground="Red", font=('times', 15, 'bold'))
        LRbutton.place(x=200, y=350)  # Placed further down for categorization

        # Button for starting Random Forest algorithm processing
        RFbutton = tk.Button(window, text="RandomForest", command=RFprocess, fg=fgcolor, bg=bgcolor1, width=14, height=1, activebackground="Red", font=('times', 15, 'bold'))
        RFbutton.place(x=200, y=450)  # Sequentially organized for clear navigation

        # Button to execute Decision Tree model processing
        SVM1button = tk.Button(window, text="Decision Tree", command=DTprocess, fg=fgcolor, bg=bgcolor1, width=16, height=1, activebackground="Red", font=('times', 15, 'bold'))
        SVM1button.place(x=500, y=250)  # A different column for clarity and UI balance

        # Button to initiate Gradient Boosting process
        PRbutton1 = tk.Button(window, text="GB", command=GBprocess, fg=fgcolor, bg=bgcolor1, width=14, height=1, activebackground="Red", font=('times', 15, 'bold'))
        PRbutton1.place(x=500, y=350)  # Same column as Decision Tree for consistency

        # Button for processing with Light Gradient Boosting Machine
        RFbutton1 = tk.Button(window, text="LGBM", command=LGBMprocess, fg=fgcolor, bg=bgcolor1, width=14, height=1, activebackground="Red", font=('times', 15, 'bold'))
        RFbutton1.place(x=500, y=450)  # Continues the logical arrangement of ML model buttons

        # Button to use the K Nearest Neighbors algorithm
        SVM1button2 = tk.Button(window, text="KNN", command=KNNprocess, fg=fgcolor, bg=bgcolor1, width=16, height=1, activebackground="Red", font=('times', 15, 'bold'))
        SVM1button2.place(x=500, y=550)  # Adds to the range of models available in this column

        # Button to use a Voting Classifier model
        SVM1button1 = tk.Button(window, text="Voting Classifier", command=VCprocess, fg=fgcolor, bg=bgcolor1, width=16, height=1, activebackground="Red", font=('times', 15, 'bold'))
        SVM1button1.place(x=800, y=250)  # Positioned to begin another column, suggesting advanced options

        # Button to trigger the prediction process based on user inputs
        SVM1button3 = tk.Button(window, text="Predict", command=Predictprocess, fg=fgcolor, bg=bgcolor1, width=16, height=1, activebackground="Red", font=('times', 15, 'bold'))
        SVM1button3.place(x=800, y=350)  # Directly under the Voting Classifier for workflow logic

        # Button to close the application
        quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg=fgcolor, bg=bgcolor1, width=15, height=1, activebackground="Red", font=('times', 15, 'bold'))
        quitWindow.place(x=800, y=450)  # Easily accessible for exiting the program

        
        # Start the GUI event loop
        window.mainloop()
Home() # Call the function to run the application

