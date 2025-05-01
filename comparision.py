import sys
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
def process():
    colors = ["#FF0000", "#1f77b4","#008000"]
    explode = (0.1, 0, 0, 0, 0)  
    
    
    alc = ["Navie Bayes","SGD Classifier", "Voting Classifier"]
    acc = [96.6819, 97.5973,97.5114]
    
    
    fig = plt.figure(0)    
    plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy %')
    plt.title('Accuracy comparision')
    plt.savefig("results/comparision.png") 
    plt.pause(5)
    plt.show(block=False)
    plt.close()
process()
