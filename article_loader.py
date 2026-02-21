import arxiv
import pandas as pd

categories = {
    "cs.CL": "NLP",
    "cs.CV": "Computer Vision",
    "cs.RO": "Robotics",
    "cs.CR": "Cybersecurity",
    "quant-ph": "Quantum Physics",
    "astro-ph": "Astrophysics",
    "q-bio": "Quantitative Biology",
    "econ.EM": "Econometrics"
}

abstracts = []
for cat, label in categories.items():
    print(f"Fetching {label}...")
    search = arxiv.Search(
        query=f"cat:{cat}",
        max_results=200
    )
    for result in search.results():
        abstracts.append({
            "title": result.title,
            "abstract": result.summary,
            "category": cat,
            "label": label
        })

df = pd.DataFrame(abstracts)
df.to_csv("arxiv_dataset.csv", index=False)
print(f"Dataset saved: {len(df)} articles")