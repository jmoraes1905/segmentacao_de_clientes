import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

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

plot_perc('pagamento_tipo', dados)

df_olist= dados[['id_unico_cliente','id_cliente','horario_pedido','item_id','preco']]

#Agrupando pelo id cliente e pegando as datas dos pedidos para calcular a recencia
df_compra = dados.groupby('id_unico_cliente').horario_pedido.max().reset_index()
df_compra.columns=['id_unico_cliente', 'DataMaxCompra']
df_compra['DataMaxCompra'] = pd.to_datetime(df_compra['DataMaxCompra'])

#Calcula recencia
df_compra['Recencia'] = (df_compra['DataMaxCompra'].max()-df_compra['DataMaxCompra']).dt.days

df_usuario = pd.merge(df_olist,df_compra[['id_unico_cliente','Recencia']],on='id_unico_cliente')

# Utilizando o metodo de elbow para clusterizar com base na recencia
from sklearn.cluster import KMeans

def calcular_wcss(data):
    elbow=[]
    for k in range(1,10):
        kmeans = KMeans(n_clusters=k, init='random',random_state=42, max_iter=300)
        kmeans.fit(data)
        data['Cluster'] = kmeans.labels_
        elbow.append(kmeans.inertia_ )
    return elbow

# KMeans espera um array-like 2d então tem que passar um dataframe não uma series
df_recencia= df_usuario[['Recencia']]

elbow = calcular_wcss(df_recencia)

plt.figure(figsize=(10,5))
plt.plot(elbow)
plt.xlabel('Numero de clusters')
plt.show()

# Numero otimo de clusters

import math

def numero_otimo_clusters(elbow):
    x0, y0 = 1, elbow[0]
    x1, y1 = len(elbow),elbow[len(elbow)-1]
    
    distancia =[]
    for i in range(len(elbow)):
        x = i+1
        y= elbow[i]
        
        numerador = abs((y1-y0)*x -(x1-x0)*y + x1*y0 -y1*x0)
        denominador = math.sqrt((y1-y0)**2 +(x1-x0)**2)
        
        distancia.append(numerador/denominador) 
        
    return distancia.index(max(distancia))+1

n_clusters = numero_otimo_clusters(elbow)

# Clusterizaçao com base na recencia
kmeans =KMeans(n_clusters=n_clusters)
df_usuario['RecenciaCluster']=kmeans.fit_predict(df_recencia)

# Atribuicao de notas: maior nota -> menor recencia -> menor label
# Reatribuir labels de maneira ordenada

def ordenador_cluster(cluster_name,target_name,df):
    grouped_by_cluster = df_usuario.groupby(cluster_name)[target_name].mean().reset_index()
    grouped_by_cluster_ordered = grouped_by_cluster.sort_values(by=target_name,ascending=False).reset_index(drop=True)
    grouped_by_cluster_ordered['index']=grouped_by_cluster_ordered.index
    joining_cluster = pd.merge(df,grouped_by_cluster_ordered[[cluster_name,'index']],on=cluster_name)
    removing_data = joining_cluster.drop([cluster_name],axis=1)
    df_final = removing_data.rename(columns={'index':cluster_name})
    return df_final

df_usuario=ordenador_cluster('RecenciaCluster', 'Recencia', df_usuario)
        
