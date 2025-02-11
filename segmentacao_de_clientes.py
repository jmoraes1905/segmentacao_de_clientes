import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

dados = pd.read_csv('BancoDeDados.csv')

dados.info()

def plot_perc(st,dados):
    plt.figure(figsize=(20,8))
    
    g=sns.countplot(x=st,data=dados,orient='h')
    g.set_ylabel('Contagem', fontsize=17)
    
    sizes=[]
    for p in g.patches:
        height=p.get_height()
        sizes.append(height)
        g.text(p.get_x() + p.get_width()/1.6,
              height+200,
              '{:1.2f}%'.format(height/116581*100),
               ha = 'center' , va='bottom',fontsize=12)
        
        g.set_ylim(0,max(sizes))
        

plot_perc('estado_cliente',dados)

plot_perc('estado_vendedor',dados)