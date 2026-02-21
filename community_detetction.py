import networkx as nx
import pandas as pd
from pathlib import Path
from community import best_partition  # Louvain
from networkx.algorithms.community import label_propagation_communities
from sklearn.cluster import SpectralClustering
import numpy as np

# Chemins
BASE_DIR = Path(__file__).parent
GRAPH_PATH = BASE_DIR / 'graph_results' / 'graph.gexf'
DATA_PATH = BASE_DIR / 'data' / 'arxiv_dataset.csv'
RESULT_DIR = BASE_DIR / 'community_results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Chargement du graphe
try:
    G = nx.read_gexf(GRAPH_PATH)
except FileNotFoundError as e:
    print(f'Graphe introuvable : {e}')
    exit()

# Chargement des articles
try:
    articles = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    print(f'Dataset introuvable : {e}')
    exit()

print(f"Graphe chargé : {G.number_of_nodes()} noeuds, {G.number_of_edges()} aretes")

# ── 1. LOUVAIN ──────────────────────────────────────────────────────────────
print("\n[1/3] Louvain...")
louvain_partition = best_partition(G)
louvain_labels = [louvain_partition[n] for n in G.nodes()]
print(f"  Communautés détectées : {len(set(louvain_labels))}")

# ── 2. LABEL PROPAGATION ─────────────────────────────────────────────────────
print("\n[2/3] Label Propagation...")
lp_communities = label_propagation_communities(G)
lp_partition = {}
for community_id, community in enumerate(lp_communities):
    for node in community:
        lp_partition[node] = community_id
lp_labels = [lp_partition[n] for n in G.nodes()]
print(f"  Communautés détectées : {len(set(lp_labels))}")

# ── 3. SPECTRAL CLUSTERING ───────────────────────────────────────────────────
print("\n[3/3] Spectral Clustering...")
adjacency_matrix = nx.to_numpy_array(G)
sc = SpectralClustering(n_clusters=8, affinity='precomputed', random_state=42)
sc_labels = sc.fit_predict(adjacency_matrix)
print(f"  Communautés détectées : {len(set(sc_labels))}")

# ── SAUVEGARDE ───────────────────────────────────────────────────────────────
results = pd.DataFrame({
    'true_label': articles['label'].values,
    'louvain': louvain_labels,
    'label_propagation': lp_labels,
    'spectral_clustering': sc_labels
})

results.to_csv(RESULT_DIR / 'community_results.csv', index=False)
print(f"\nRésultats sauvegardés dans community_results/community_results.csv")