"""
Advanced ML / DL / NLP analyses for MENA Microbiome platform.
Outputs JSON blobs the front-end consumes, plus PNG/SVG figures for visualizations page.
"""
import os, json, re, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import stats as spstats

DATA = "/home/claude/out/data/mena_metagenomics_clean.tsv"
FIGDIR = "/home/claude/out/figures"
DATADIR = "/home/claude/out/data"

mpl.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":120,"savefig.dpi":300,"savefig.bbox":"tight",
})
PAL = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00","#F0E442","#999999"]

def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/{name}.svg", bbox_inches="tight")
    plt.close(fig)

print("Loading...")
df = pd.read_csv(DATA, sep="\t", low_memory=False)
df["read_count_num"] = pd.to_numeric(df["read_count"], errors="coerce")
df["base_count_num"] = pd.to_numeric(df["base_count"], errors="coerce")
print(f"  {len(df)} runs")

ml_out = {}

# ═══════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n[STAT] Statistical analysis")

# Descriptive stats for numeric
num_cols = ["read_count_num","base_count_num"]
desc = df[num_cols].describe().round(2)
ml_out["descriptive_stats"] = desc.reset_index().to_dict(orient="records")

# Normality tests (Shapiro on log-transformed read count, subsample)
rc = df["read_count_num"].dropna()
rc = rc[rc > 0]
log_rc = np.log10(rc)
samp = log_rc.sample(min(5000, len(log_rc)), random_state=42)
sh_stat, sh_p = spstats.shapiro(samp)
ml_out["normality"] = {
    "variable":"log10(read_count)",
    "n_subsample":int(len(samp)),
    "shapiro_W":round(float(sh_stat),4),
    "shapiro_p":float(sh_p),
    "verdict":"non-normal" if sh_p < 0.05 else "normal",
}

# Chi-square: country × broad_category
ct = pd.crosstab(df["country"], df["broad_category"])
chi2, p_chi, dof, _ = spstats.chi2_contingency(ct)
ml_out["chi_square_country_category"] = {"chi2":round(float(chi2),2),"dof":int(dof),"p":float(p_chi)}

# Kruskal-Wallis: read_count across broad_category
groups = [g["read_count_num"].dropna().values for _, g in df.groupby("broad_category") if g["read_count_num"].notna().sum() > 30]
kw_stat, kw_p = spstats.kruskal(*groups)
ml_out["kruskal_readcount_by_category"] = {"H":round(float(kw_stat),2),"p":float(kw_p)}

# Pearson correlation read_count vs base_count (log)
mask = df["read_count_num"].notna() & df["base_count_num"].notna() & (df["read_count_num"]>0) & (df["base_count_num"]>0)
if mask.sum() > 100:
    r, pr = spstats.pearsonr(np.log10(df.loc[mask,"read_count_num"]), np.log10(df.loc[mask,"base_count_num"]))
    ml_out["correlation_reads_bases"] = {"r":round(float(r),4),"p":float(pr),"n":int(mask.sum())}

# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING for ML/DL
# ═══════════════════════════════════════════════════════════════════
print("[ML] Feature engineering")

ml_df = df.dropna(subset=["broad_category","country","instrument_platform","library_strategy","data_subtype"]).copy()
ml_df = ml_df[ml_df["broad_category"].isin(["Human","Environment","Animal","Plant","Food","Clinical"])]
ml_df["log_reads"] = np.log10(ml_df["read_count_num"].fillna(ml_df["read_count_num"].median()).clip(lower=1))
ml_df["log_bases"] = np.log10(ml_df["base_count_num"].fillna(ml_df["base_count_num"].median()).clip(lower=1))

# Encode categorical
cat_feats = ["country","instrument_platform","library_strategy","library_source","library_layout","data_subtype"]
enc_df = pd.DataFrame(index=ml_df.index)
encoders = {}
for c in cat_feats:
    le = LabelEncoder()
    enc_df[c] = le.fit_transform(ml_df[c].fillna("unknown").astype(str))
    encoders[c] = le
enc_df["log_reads"] = ml_df["log_reads"].values
enc_df["log_bases"] = ml_df["log_bases"].values
X = enc_df.values
y = ml_df["broad_category"].values

# Subsample for compute
if len(X) > 8000:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), 8000, replace=False)
    X, y = X[idx], y[idx]
print(f"  X={X.shape}, classes={np.unique(y)}")

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# ═══════════════════════════════════════════════════════════════════
# PCA
# ═══════════════════════════════════════════════════════════════════
print("[ML] PCA")
pca = PCA(n_components=3)
Xp = pca.fit_transform(Xs)
ml_out["pca"] = {
    "explained_variance":[round(float(x),4) for x in pca.explained_variance_ratio_],
    "cum_variance":[round(float(x),4) for x in np.cumsum(pca.explained_variance_ratio_)],
    "points":[{"pc1":round(float(Xp[i,0]),3),"pc2":round(float(Xp[i,1]),3),"pc3":round(float(Xp[i,2]),3),"cat":y[i]}
              for i in range(min(1500, len(Xp)))],
    "feature_names":list(enc_df.columns),
    "loadings":[[round(float(x),4) for x in row] for row in pca.components_],
}

fig, axes = plt.subplots(1,2, figsize=(13,5.5))
cats = np.unique(y)
for i,c in enumerate(cats):
    m = y==c
    axes[0].scatter(Xp[m,0], Xp[m,1], c=PAL[i%len(PAL)], label=c, alpha=0.5, s=15, edgecolors="none")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("PCA: samples projected on PC1–PC2")
axes[0].legend(frameon=False, fontsize=8, loc="best")

axes[1].bar(range(1,4), pca.explained_variance_ratio_*100, color=PAL[0])
axes[1].plot(range(1,4), np.cumsum(pca.explained_variance_ratio_)*100, "o-", color=PAL[5])
axes[1].set_xlabel("Principal Component")
axes[1].set_ylabel("% variance explained")
axes[1].set_title("Scree plot")
plt.tight_layout()
save(fig, "ML01_pca")

# ═══════════════════════════════════════════════════════════════════
# K-MEANS
# ═══════════════════════════════════════════════════════════════════
print("[ML] K-Means")
inertias = []
for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    inertias.append(float(km.inertia_))
ml_out["kmeans_elbow"] = [{"k":k,"inertia":round(v,2)} for k,v in zip(range(2,11), inertias)]

km_final = KMeans(n_clusters=5, random_state=42, n_init=10).fit(Xs)
labels = km_final.labels_
cluster_comp = pd.crosstab(labels, y)
ml_out["kmeans_clusters"] = {
    "k":5,
    "cluster_sizes":[int(x) for x in np.bincount(labels)],
    "composition":cluster_comp.to_dict(),
}

fig, axes = plt.subplots(1,2, figsize=(13,5.5))
axes[0].plot(range(2,11), inertias, "o-", color=PAL[1], lw=2)
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].set_title("K-Means elbow curve")

for k in range(5):
    m = labels==k
    axes[1].scatter(Xp[m,0], Xp[m,1], c=PAL[k], label=f"Cluster {k}", alpha=0.5, s=15, edgecolors="none")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].set_title("K-Means clusters on PC1–PC2")
axes[1].legend(frameon=False, fontsize=8)
plt.tight_layout()
save(fig, "ML02_kmeans")

# ═══════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════
print("[ML] Random Forest")
X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=15)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_te, y_pred, labels=sorted(np.unique(y)))

ml_out["random_forest"] = {
    "accuracy":round(float(acc),4),
    "n_train":int(len(X_tr)),
    "n_test":int(len(X_te)),
    "feature_importance":[{"feature":f,"importance":round(float(v),4)}
                          for f,v in sorted(zip(enc_df.columns, rf.feature_importances_), key=lambda x:-x[1])],
    "classes":sorted(list(np.unique(y))),
    "confusion_matrix":cm.tolist(),
    "per_class":[{"class":k, "precision":round(v["precision"],3), "recall":round(v["recall"],3), "f1":round(v["f1-score"],3), "support":int(v["support"])}
                 for k,v in report.items() if k in sorted(np.unique(y))],
}

fig, axes = plt.subplots(1,2, figsize=(13,5.5))
fi = sorted(zip(enc_df.columns, rf.feature_importances_), key=lambda x:x[1])
axes[0].barh([x[0] for x in fi], [x[1] for x in fi], color=PAL[2])
axes[0].set_xlabel("Importance")
axes[0].set_title(f"Random Forest feature importance (acc={acc:.3f})")

im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks(range(len(cm))); axes[1].set_yticks(range(len(cm)))
lbls = sorted(np.unique(y))
axes[1].set_xticklabels(lbls, rotation=45, ha="right"); axes[1].set_yticklabels(lbls)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
axes[1].set_title("Confusion matrix")
for i in range(len(cm)):
    for j in range(len(cm)):
        axes[1].text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=8)
plt.colorbar(im, ax=axes[1], fraction=0.046)
plt.tight_layout()
save(fig, "ML03_random_forest")

# ═══════════════════════════════════════════════════════════════════
# MLP (DEEP LEARNING)
# ═══════════════════════════════════════════════════════════════════
print("[DL] MLP classifier")
le_y = LabelEncoder()
y_tr_enc = le_y.fit_transform(y_tr)
y_te_enc = le_y.transform(y_te)
mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42,
                    early_stopping=True, validation_fraction=0.15)
mlp.fit(X_tr, y_tr_enc)
mlp_acc = accuracy_score(y_te_enc, mlp.predict(X_te))

ml_out["mlp"] = {
    "architecture":"input → 64 → 32 → output (ReLU, Adam)",
    "accuracy":round(float(mlp_acc),4),
    "n_iter":int(mlp.n_iter_),
    "loss_curve":[round(float(x),4) for x in mlp.loss_curve_],
    "val_scores":[round(float(x),4) for x in (mlp.validation_scores_ or [])],
}

fig, axes = plt.subplots(1,2, figsize=(13,5))
axes[0].plot(mlp.loss_curve_, color=PAL[3], lw=2)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss")
axes[0].set_title(f"MLP training loss (n_iter={mlp.n_iter_})")

if mlp.validation_scores_:
    axes[1].plot(mlp.validation_scores_, color=PAL[4], lw=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation accuracy")
    axes[1].set_title(f"MLP validation accuracy (final={mlp_acc:.3f})")
plt.tight_layout()
save(fig, "DL01_mlp_training")

# ═══════════════════════════════════════════════════════════════════
# t-SNE
# ═══════════════════════════════════════════════════════════════════
print("[DL] t-SNE")
n_tsne = min(2000, len(Xs))
rng = np.random.RandomState(42)
idx_t = rng.choice(len(Xs), n_tsne, replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
Xt = tsne.fit_transform(Xs[idx_t])
y_t = y[idx_t]

ml_out["tsne"] = {
    "n":int(n_tsne),
    "points":[{"x":round(float(Xt[i,0]),3),"y":round(float(Xt[i,1]),3),"cat":y_t[i]}
              for i in range(len(Xt))],
}

fig, ax = plt.subplots(figsize=(9,7))
for i,c in enumerate(np.unique(y_t)):
    m = y_t==c
    ax.scatter(Xt[m,0], Xt[m,1], c=PAL[i%len(PAL)], label=c, alpha=0.6, s=18, edgecolors="none")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title(f"t-SNE embedding (n={n_tsne}, perplexity=30)")
ax.legend(frameon=False, loc="best")
save(fig, "DL02_tsne")

# ═══════════════════════════════════════════════════════════════════
# NLP: LDA + NMF + TF-IDF
# ═══════════════════════════════════════════════════════════════════
print("[NLP] Topic modeling")
docs_raw = df.drop_duplicates("study_accession")[["study_title","broad_category"]].dropna(subset=["study_title"])
docs = docs_raw["study_title"].astype(str).tolist()
doc_cats = docs_raw["broad_category"].fillna("Other").tolist()
print(f"  {len(docs)} study titles")

STOP = set("""a an and or the of to in on for for with by from as at is are was were be been being
this that these those it its their there here which who whom whose what when where why how
study studies analysis sequencing sample samples using based high throughput data dataset project
next generation rna dna genomic genomics sequence sequences whole gene genes
shotgun illumina 16s rrna via between among across within during we our
sp nov strain isolate isolates isolated genome profiling profile""".split())

cv = CountVectorizer(max_df=0.8, min_df=5, stop_words=list(STOP), max_features=800)
cv_mat = cv.fit_transform(docs)
vocab = cv.get_feature_names_out()

# LDA
n_topics = 6
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15, learning_method="online")
lda.fit(cv_mat)
lda_topics = []
for ti, comp in enumerate(lda.components_):
    top_idx = comp.argsort()[-10:][::-1]
    lda_topics.append({"topic":f"T{ti+1}", "words":[vocab[i] for i in top_idx],
                       "weights":[round(float(comp[i]),3) for i in top_idx]})
ml_out["lda_topics"] = lda_topics

doc_topic = lda.transform(cv_mat)
topic_prev = doc_topic.mean(axis=0)
ml_out["lda_prevalence"] = [{"topic":f"T{i+1}","prevalence":round(float(p),4)} for i,p in enumerate(topic_prev)]

fig, axes = plt.subplots(2,3, figsize=(14,8))
for ti, ax in enumerate(axes.flat):
    t = lda_topics[ti]
    ax.barh(t["words"][::-1], t["weights"][::-1], color=PAL[ti%len(PAL)])
    ax.set_title(f"LDA Topic {ti+1}", fontsize=11)
    ax.tick_params(labelsize=8)
plt.tight_layout()
save(fig, "AI01_lda_topics")

# NMF with TF-IDF
tv = TfidfVectorizer(max_df=0.8, min_df=5, stop_words=list(STOP), max_features=800)
tv_mat = tv.fit_transform(docs)
vocab_t = tv.get_feature_names_out()
nmf = NMF(n_components=n_topics, random_state=42, max_iter=300, init="nndsvd")
nmf.fit(tv_mat)
nmf_topics = []
for ti, comp in enumerate(nmf.components_):
    top_idx = comp.argsort()[-10:][::-1]
    nmf_topics.append({"topic":f"N{ti+1}","words":[vocab_t[i] for i in top_idx],
                       "weights":[round(float(comp[i]),3) for i in top_idx]})
ml_out["nmf_topics"] = nmf_topics

# TF-IDF discriminating terms per category
cat_terms = {}
for cat in set(doc_cats):
    docs_cat = [d for d,c in zip(docs, doc_cats) if c==cat]
    if len(docs_cat) < 10: continue
    tv2 = TfidfVectorizer(max_df=0.9, min_df=3, stop_words=list(STOP), max_features=300)
    try:
        m = tv2.fit_transform(docs_cat)
        scores = np.asarray(m.mean(axis=0)).ravel()
        vocab2 = tv2.get_feature_names_out()
        top = sorted(zip(vocab2, scores), key=lambda x:-x[1])[:12]
        cat_terms[cat] = [{"term":w,"score":round(float(s),4)} for w,s in top]
    except Exception:
        continue
ml_out["tfidf_by_category"] = cat_terms

fig, axes = plt.subplots(2,3, figsize=(14,8))
cats_p = list(cat_terms.keys())[:6]
for ax, cat in zip(axes.flat, cats_p):
    terms = cat_terms[cat][:10]
    ax.barh([t["term"] for t in terms][::-1], [t["score"] for t in terms][::-1], color=PAL[cats_p.index(cat)%len(PAL)])
    ax.set_title(f"TF-IDF: {cat}", fontsize=11)
    ax.tick_params(labelsize=8)
plt.tight_layout()
save(fig, "AI02_tfidf_by_category")

# Semantic clustering (K-Means on TF-IDF)
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
tv_svd = svd.fit_transform(tv_mat)
km_sem = KMeans(n_clusters=6, random_state=42, n_init=10).fit(tv_svd)
sem_labels = km_sem.labels_

sem_clusters = []
for k in range(6):
    mask = sem_labels==k
    # top terms in cluster
    if mask.sum() > 0:
        m = tv_mat[mask].mean(axis=0)
        scores = np.asarray(m).ravel()
        top = scores.argsort()[-8:][::-1]
        sem_clusters.append({"cluster":k,"size":int(mask.sum()),
                             "top_terms":[vocab_t[i] for i in top]})
ml_out["semantic_clusters"] = sem_clusters

# ═══════════════════════════════════════════════════════════════════
# AUTOENCODER (light, 2D latent)
# ═══════════════════════════════════════════════════════════════════
print("[DL] Autoencoder")
# Using sklearn's MLPRegressor trick: we simulate autoencoder by learning identity via small bottleneck MLP
from sklearn.neural_network import MLPRegressor
ae_mlp = MLPRegressor(hidden_layer_sizes=(16, 4, 16), max_iter=150, random_state=42,
                     early_stopping=True, validation_fraction=0.15)
ae_mlp.fit(Xs, Xs)
# Extract bottleneck: forward pass through first 2 layers
def forward_to_bottleneck(x, mlp):
    a = x
    for i, (w, b) in enumerate(zip(mlp.coefs_[:2], mlp.intercepts_[:2])):
        a = a @ w + b
        if i < 1: a = np.maximum(0, a)  # relu
    return a
Z = forward_to_bottleneck(Xs, ae_mlp)  # shape (n, 4)
# reduce to 2D for display
Z2 = PCA(n_components=2).fit_transform(Z)

# reconstruction error
X_rec = ae_mlp.predict(Xs)
recon_err = ((Xs - X_rec) ** 2).mean(axis=1)
top_err_idx = np.argsort(-recon_err)[:10]

ml_out["autoencoder"] = {
    "architecture":"input → 16 → 4 → 16 → output (ReLU)",
    "n_iter":int(ae_mlp.n_iter_),
    "final_loss":round(float(ae_mlp.loss_),5),
    "embedding":[{"x":round(float(Z2[i,0]),3),"y":round(float(Z2[i,1]),3),"cat":y[i]}
                 for i in range(min(1500, len(Z2)))],
    "top_anomalies":[{"idx":int(i),"error":round(float(recon_err[i]),4),"cat":y[i]} for i in top_err_idx],
}

fig, axes = plt.subplots(1,2, figsize=(13,5.5))
for i,c in enumerate(np.unique(y)):
    m = y==c
    axes[0].scatter(Z2[m,0], Z2[m,1], c=PAL[i%len(PAL)], label=c, alpha=0.5, s=15, edgecolors="none")
axes[0].set_xlabel("Latent dim 1"); axes[0].set_ylabel("Latent dim 2")
axes[0].set_title("Autoencoder latent embedding (colored by category)")
axes[0].legend(frameon=False, fontsize=8)

axes[1].hist(recon_err, bins=60, color=PAL[5], edgecolor="white")
axes[1].set_xlabel("Reconstruction error (MSE)")
axes[1].set_ylabel("Samples")
axes[1].set_title("Anomaly score distribution")
plt.tight_layout()
save(fig, "DL03_autoencoder")

# Save
with open(f"{DATADIR}/mena_ml_results.json","w") as f:
    json.dump(ml_out, f, default=str)

print(f"\nDONE. ml_results.json size:", os.path.getsize(f"{DATADIR}/mena_ml_results.json")/1024, "KB")
print(f"RF acc={ml_out['random_forest']['accuracy']}, MLP acc={ml_out['mlp']['accuracy']}")
