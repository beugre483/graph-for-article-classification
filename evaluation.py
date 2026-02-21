import pandas as pd
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

# Chemins
BASE_DIR = Path(__file__).parent
RESULT_PATH = BASE_DIR / 'community_results' / 'community_results.csv'

# Chargement des résultats
try:
    df = pd.read_csv(RESULT_PATH)
except FileNotFoundError as e:
    print(f'Fichier résultats introuvable : {e}')
    exit()

# Encoder les vrais labels en chiffres (NLP → 0, Astrophysics → 1, etc.)
le = LabelEncoder()
true_labels = le.fit_transform(df['true_label'])

# ── CALCUL NMI ET ARI ────────────────────────────────────────────────────────
algorithmes = ['louvain', 'label_propagation', 'spectral_clustering']

print("=" * 50)
print(f"{'Algorithme':<25} {'NMI':>8} {'ARI':>8}")
print("=" * 50)

resultats = []
for algo in algorithmes:
    predicted = df[algo].values
    nmi = normalized_mutual_info_score(true_labels, predicted)
    ari = adjusted_rand_score(true_labels, predicted)
    resultats.append({'algorithme': algo, 'NMI': round(nmi, 4), 'ARI': round(ari, 4)})
    print(f"{algo:<25} {nmi:>8.4f} {ari:>8.4f}")

print("=" * 50)

# Meilleur algorithme
best = max(resultats, key=lambda x: x['NMI'])
print(f"\n✅ Meilleur algorithme (NMI) : {best['algorithme']} ({best['NMI']})")

# Sauvegarde
resultats_df = pd.DataFrame(resultats)
resultats_df.to_csv(BASE_DIR / 'community_results' / 'evaluation.csv', index=False)
print(f"\nRésultats sauvegardés dans community_results/evaluation.csv")