import numpy as np
import pandas as pd 
import networkx as nx
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
#chemin
BASE_DIR=Path(__file__).parent
RESULT_DIR=BASE_DIR/'embeddings_results'/'embeddings.npy'
GRAPH_RESULT=BASE_DIR/'graph_results'
GRAPH_RESULT.mkdir(parents=True,exist_ok=True)
Data_Dir=BASE_DIR/'data'/'arxiv_dataset.csv'

#charger les embeddings
try :
    embeddings=np.load(RESULT_DIR)
except FileNotFoundError as e:
    print(f'fichier embeddings non présents :{e}')
    exit()
#charger la dataset des articles
try:
    articles=pd.read_csv(Data_Dir)
except FileNotFoundError as e:
    print(f'data articles manquants: {e}')
    exit()
    
#appliquer le cosine similarty sur les embeddings et les stocker
similarity_matrix=cosine_similarity(embeddings)

#construction du graph avec K=10
G=nx.Graph()

#ajouter 1600 noeuds aux graphes 

G.add_nodes_from(range(len(articles)))

#construire les arretes

for i in range(len(articles)):
    row=similarity_matrix[i]
    
    sorted_indices=np.argsort(row)[::-1]
    
    top_indices=sorted_indices[1:15]
    
    for j in top_indices:
        score=row(j)
        
        G.add_edge(i,j,weight=score)

nx.write_gexf(G, GRAPH_RESULT / "graph_final.gexf")
    