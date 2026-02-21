import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os 
import numpy as np

#chemin
BASE_DIR=Path(__file__).parent
RESULT_DIR=BASE_DIR/'embeddings_results'
RESULT_DIR.mkdir(parents=True,exist_ok=True)
Data_Dir=BASE_DIR/'data'/'arxiv_dataset.csv'

#setup model
model=SentenceTransformer('all-MiniLM-L6-v2')

try:
   articles=pd.read_csv(Data_Dir)
except FileNotFoundError as e :
    print(f'error:{e}')
    

#creation serie d'abstract pour passer au model
abstracts=articles['abstract']

#variables embeddings
embeddings=model.encode(abstracts)
np.save(RESULT_DIR/"embeddings.npy", embeddings)
