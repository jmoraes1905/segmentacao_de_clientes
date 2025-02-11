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

# Funcao recebe o label do atributo, o atributo clusterizado, o banco de dados e o critétio para pontuação
## Para recencia, quanto menor maior a pontuação, para a frequencia, o contrario
def ordenador_cluster(cluster_name,target_name,df, ascending):
    grouped_by_cluster = df_usuario.groupby(cluster_name)[target_name].mean().reset_index()
    grouped_by_cluster_ordered = grouped_by_cluster.sort_values(by=target_name,ascending=ascending).reset_index(drop=True)
    grouped_by_cluster_ordered['index']=grouped_by_cluster_ordered.index
    joining_cluster = pd.merge(df,grouped_by_cluster_ordered[[cluster_name,'index']],on=cluster_name)
    removing_data = joining_cluster.drop([cluster_name],axis=1)
    df_final = removing_data.rename(columns={'index':cluster_name})
    return df_final

df_usuario=ordenador_cluster('RecenciaCluster', 'Recencia', df_usuario, False)

# Seguimos a mesma pipeline para a frequencia:
# Agrupa por id do cliente e pega a frequencia de compra, calcula a frequencia, faz o merge com o banco de dados
# Roda a clusterizaçao na frequencia e faz ordenaçaõ para pontuar

df_frequencia = dados.groupby('id_unico_cliente').pedido_aprovado.count().reset_index()
df_frequencia.columns = ['id_unico_cliente','Frequencia']

df_usuario = pd.merge(df_usuario,df_frequencia,on='id_unico_cliente')

df_frequencia = df_usuario[['Frequencia']]
df_usuario['FrequenciaCluster']=kmeans.fit_predict(df_frequencia)

df_usuario=ordenador_cluster('FrequenciaCluster', 'Frequencia', df_usuario, True)

#df_usuario.groupby('FrequenciaCluster')['Frequencia'].describe()

#Seguimos o mesmo procedimento para a receita
df_receita = dados.groupby('id_unico_cliente').pagamento_valor.sum().reset_index()
df_receita.columns=['id_unico_cliente','Receita']

df_usuario = pd.merge(df_usuario,df_receita,on='id_unico_cliente')

df_receita= df_usuario[['Receita']]
df_usuario['ReceitaCluster']=kmeans.fit_predict(df_receita)

df_usuario= ordenador_cluster("ReceitaCluster", "Receita", df_usuario, True)

#df_usuario.groupby('ReceitaCluster')['Receita'].describe()

# Atribuindo as pontuações em um novo banco

df_final = df_usuario[['id_unico_cliente', 'Recencia','RecenciaCluster','Frequencia','FrequenciaCluster','Receita','ReceitaCluster']]

df_final['Pontuacao'] = df_final['RecenciaCluster'] + df_final['FrequenciaCluster'] + df_final['ReceitaCluster']

df_final['Segmento'] = 'Inativo'

df_final.loc[df_final['Pontuacao']>=1,'Segmento'] ='Business'
df_final.loc[df_final['Pontuacao']>=3,'Segmento'] ='Master'
df_final.loc[df_final['Pontuacao']>=5,'Segmento'] ='Premium'

#df_final['Segmento'].value_counts()

df_final.to_csv('RFM.csv')

# Procedendo para analise visual: features analisadas duas a duas em scatterplot

def plot_segmento(x,y,data):
    sns.set(palette='muted',color_codes=True,style='whitegrid')
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=x,y=y,hue='Segmento',data=data,size='Segmento',sizes=(50,150),size_order=['Premium','Master','Business','Inativo'])
    plt.show()

plot_segmento('Recencia', 'Frequencia', df_final)

plot_segmento('Frequencia', 'Receita', df_final)

plot_segmento('Receita', 'Recencia', df_final)

#sns.countplot(df_final['Segmento'])
plot_perc('Segmento', df_final)