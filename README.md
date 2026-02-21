# 📄 Graph-Based Article Classification

Détection de communautés thématiques dans un corpus d'articles scientifiques arXiv à l'aide d'algorithmes de graphes.

---

## 🎯 Objectif

L'objectif de ce projet est de répondre à la question suivante :

> **Est-ce qu'un algorithme de graphe peut retrouver automatiquement les thématiques d'articles scientifiques, sans jamais avoir vu les labels ?**

Pour cela, on construit un graphe de similarité entre articles, on applique des algorithmes de détection de communautés, et on évalue les résultats en les comparant aux vrais labels arXiv.

---

## 📁 Structure du projet

```
graph-for-article-classifi/
│
├── data/
│   └── arxiv_dataset.csv          # Dataset collecté depuis arXiv
│
├── embeddings_results/
│   └── embeddings.npy             # Embeddings générés par Sentence-BERT
│
├── graph_results/
│   └── graph.gexf                 # Graphe de similarité (format Gephi)
│
├── community_results/
│   ├── community_results.csv      # Résultats des 3 algorithmes
│   └── evaluation.csv             # Scores NMI et ARI
│
├── figures/                       # Visualisations et graphiques
│
├── 1_articles_loader.py              # Collecte des articles depuis arXiv
├── 2_embeddings.py                # Génération des embeddings (Sentence-BERT)
├── 3_build_graph.py               # Construction du graphe KNN
├── 4_community_detection.py       # Détection de communautés
├── 5_evaluation.py                # Évaluation et comparaison des algorithmes
│
└── README.md
```

---

## 🗂️ Dataset

- **Source** : API arXiv
- **Taille** : ~1600 articles (200 par thématique)
- **8 thématiques** :

| Catégorie arXiv | Domaine |
|---|---|
| cs.CL | NLP |
| cs.CV | Computer Vision |
| cs.RO | Robotics |
| cs.CR | Cybersecurity |
| quant-ph | Quantum Physics |
| astro-ph | Astrophysics |
| q-bio | Quantitative Biology |
| econ.EM | Econometrics |

---

## ⚙️ Méthodologie

### 1. Collecte des données
Récupération des abstracts depuis l'API arXiv pour 8 catégories distinctes.

### 2. Génération des embeddings
Conversion des abstracts en vecteurs numériques avec le modèle **Sentence-BERT** (`all-MiniLM-L6-v2`).

### 3. Construction du graphe
- Calcul de la **similarité cosine** entre tous les articles
- Construction d'un graphe **KNN (k=15)** : chaque article est relié à ses 10 articles les plus similaires
- Le poids de chaque arête = score de similarité cosine

### 4. Détection de communautés
Trois algorithmes comparés :
- **Louvain** — optimisation de la modularité
- **Label Propagation** — propagation de labels par vote majoritaire
- **Spectral Clustering** — clustering basé sur les valeurs propres de la matrice d'adjacence

### 5. Évaluation
Comparaison des communautés détectées avec les vrais labels arXiv via :
- **NMI** (Normalized Mutual Information)
- **ARI** (Adjusted Rand Index)

---

## 📊 Résultats

| Algorithme | NMI | ARI | Communautés détectées |
|---|---|---|---|
| **Louvain** | **0.8033** | **0.7641** | 10 |
| Spectral Clustering | 0.7991 | 0.7154 | 8 |
| Label Propagation | 0.6815 | 0.4526 | 32 |

✅ **Louvain** obtient les meilleurs résultats avec un NMI de 0.80 et un ARI de 0.76.

---

## 🛠️ Installation

```bash
poetry install
```

---

##  Exécution

Lancer les scripts dans l'ordre :

```bash
poetry run python article_loader.py
poetry run python embeddings.py
poetry run python build_graph.py
poetry run python community_detection.py
poetry run python evaluation.py
```


---

## 📚 Références

- Sentence-BERT : [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
- Algorithme de Louvain : Blondel et al. (2008)
- API arXiv : [arxiv.org/help/api](https://arxiv.org/help/api)
