import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Similarity Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* App background */
  .stApp {
    background: #0d0f14;
    color: #e8eaf0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1e2330;
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #7c8cf8;
  }

  /* Hero header */
  .hero {
    background: linear-gradient(135deg, #1a1d2e 0%, #0f1218 60%, #0d1520 100%);
    border: 1px solid #1e2742;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(124,140,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
  }
  .hero-accent {
    color: #7c8cf8;
  }
  .hero-sub {
    color: #6b7280;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
  }
  .model-badge {
    display: inline-block;
    background: #1e2742;
    color: #7c8cf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid #2d3a6b;
    margin-top: 1rem;
  }

  /* Cards */
  .card {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #7c8cf8;
    margin-bottom: 0.75rem;
  }

  /* Score pills */
  .score-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-radius: 8px;
    background: #0d0f14;
    border: 1px solid #1e2330;
    margin-bottom: 8px;
  }
  .score-text { color: #c9cfe8; font-size: 0.9rem; }
  .score-val  { font-family: 'Space Mono', monospace; font-size: 0.85rem; font-weight: 700; }
  .score-high { color: #5ee7a0; }
  .score-mid  { color: #f5c542; }
  .score-low  { color: #f87171; }

  /* Critical thinking box */
  .ct-box {
    background: #0d111c;
    border-left: 3px solid #7c8cf8;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #b0b8d8;
    line-height: 1.6;
  }
  .ct-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #7c8cf8;
    font-weight: 700;
    margin-bottom: 3px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }

  /* Streamlit overrides */
  .stTextArea textarea, .stTextInput input {
    background: #0d0f14 !important;
    border: 1px solid #1e2330 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #7c8cf8 !important;
    box-shadow: 0 0 0 2px rgba(124,140,248,0.15) !important;
  }
  .stButton > button {
    background: #7c8cf8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  div[data-testid="stMetric"] {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 0.8rem 1rem;
  }
  div[data-testid="stMetric"] label { color: #6b7280 !important; }
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #7c8cf8 !important;
    font-family: 'Space Mono', monospace !important;
  }

  /* Section headers */
  h2 { color: #c9cfe8 !important; font-weight: 600 !important; }
  h3 { color: #9ea8cc !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_color(score):
    if score >= 0.65:
        return "score-high"
    elif score >= 0.35:
        return "score-mid"
    return "score-low"

def make_dark_layout():
    return dict(
        paper_bgcolor="#13161e",
        plot_bgcolor="#0d0f14",
        font=dict(family="Inter", color="#c9cfe8"),
        margin=dict(t=40, b=20, l=20, r=20),
    )

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("**Model**")
    st.markdown("`all-MiniLM-L6-v2`  \nSentence Transformers (free, no API key)")
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown(
        "1. Enter a **query** sentence.\n"
        "2. Add **comparison** sentences (one per line).\n"
        "3. Click **Analyse**.\n"
        "4. Explore the graphs and critical thinking notes."
    )

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">NLP <span class="hero-accent">Similarity</span> Explorer</p>
  <p class="hero-sub">Compare sentences using a free pretrained transformer — no preprocessing, no training required.</p>
  <span class="model-badge">all-MiniLM-L6-v2 · Sentence Transformers</span>
</div>
""", unsafe_allow_html=True)

# ─── Input Section ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card-title">Query Sentence</div>', unsafe_allow_html=True)
    query = st.text_input(
        label="query",
        value="Artificial intelligence is transforming the world.",
        label_visibility="collapsed",
        placeholder="Enter your main sentence here…",
    )

with col2:
    st.markdown('<div class="card-title">Comparison Sentences (one per line)</div>', unsafe_allow_html=True)
    default_sentences = (
        "Machine learning is changing every industry.\n"
        "The cat sat quietly on the mat.\n"
        "Deep learning models require a lot of data.\n"
        "Python is a popular programming language.\n"
        "Neural networks mimic the human brain.\n"
        "The weather today is quite pleasant.\n"
        "Natural language processing enables computers to understand text."
    )
    comparisons_raw = st.text_area(
        label="comparisons",
        value=default_sentences,
        height=170,
        label_visibility="collapsed",
        placeholder="One sentence per line…",
    )

run = st.button("Analyse Similarity", use_container_width=False)

# ─── Analysis ──────────────────────────────────────────────────────────────────
if run:
    sentences = [s.strip() for s in comparisons_raw.strip().splitlines() if s.strip()]
    if not sentences:
        st.warning("Please add at least one comparison sentence.")
        st.stop()
    if not query.strip():
        st.warning("Please enter a query sentence.")
        st.stop()

    with st.spinner("Loading model and computing embeddings…"):
        model = load_model()
        all_texts = [query] + sentences
        embeddings = model.encode(all_texts, show_progress_bar=False)

    query_emb = embeddings[0:1]
    comp_embs  = embeddings[1:]
    scores = cosine_similarity(query_emb, comp_embs)[0]

    # Sort by score descending
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    ranked_sentences, ranked_scores = zip(*ranked)

    st.markdown("---")

    # ── Metrics Row ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentences compared", len(sentences))
    m2.metric("Highest similarity", f"{max(scores):.3f}")
    m3.metric("Lowest similarity",  f"{min(scores):.3f}")
    m4.metric("Average similarity", f"{np.mean(scores):.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Similarity Scores List ──
    st.markdown("### Similarity Scores")
    st.markdown('<div class="card-title">Ranked by cosine similarity (1.0 = identical, 0.0 = unrelated)</div>', unsafe_allow_html=True)
    for sent, sc in zip(ranked_sentences, ranked_scores):
        cls = get_color(sc)
        st.markdown(
            f'<div class="score-row">'
            f'  <span class="score-text">{sent}</span>'
            f'  <span class="score-val {cls}">{sc:.4f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    #  GRAPH 1 — Bar Chart
    # ══════════════════════════════════════════════════
    st.markdown("### Graph 1 — Top Similar Sentences")
    bar_colors = ["#5ee7a0" if s >= 0.65 else "#f5c542" if s >= 0.35 else "#f87171"
                  for s in ranked_scores]
    fig_bar = go.Figure(go.Bar(
        x=list(ranked_scores),
        y=[f"{s[:55]}…" if len(s) > 55 else s for s in ranked_sentences],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{s:.3f}" for s in ranked_scores],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11, color="#c9cfe8"),
    ))
    fig_bar.update_layout(
        **make_dark_layout(),
        height=340,
        xaxis=dict(range=[0, 1.05], tickformat=".2f", gridcolor="#1e2330", title="Cosine Similarity"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        title=dict(text="Similarity to Query", font=dict(size=13, color="#7c8cf8"), x=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ══════════════════════════════════════════════════
    #  GRAPH 2 — Pairwise Heatmap
    # ══════════════════════════════════════════════════
    st.markdown("### Graph 2 — Pairwise Similarity Heatmap")
    labels = ["[QUERY]"] + [f"S{i+1}" for i in range(len(sentences))]
    sim_matrix = cosine_similarity(embeddings)

    fig_heat = go.Figure(go.Heatmap(
        z=sim_matrix,
        x=labels, y=labels,
        colorscale=[[0, "#0d0f14"], [0.5, "#2d3a6b"], [1, "#7c8cf8"]],
        zmin=0, zmax=1,
        text=np.round(sim_matrix, 2),
        texttemplate="%{text}",
        textfont=dict(size=9, family="Space Mono"),
        showscale=True,
    ))
    fig_heat.update_layout(
        **make_dark_layout(),
        height=420,
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        title=dict(text="Pairwise Cosine Similarity — all sentences", font=dict(size=13, color="#7c8cf8"), x=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ══════════════════════════════════════════════════
    #  GRAPH 3 — PCA 2-D Embedding Plot
    # ══════════════════════════════════════════════════
    st.markdown("### Graph 3 — 2-D Embedding Plot (PCA)")
    if len(all_texts) >= 2:
        n_comp = min(2, len(all_texts))
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(embeddings)
        if n_comp == 1:
            coords = np.column_stack([coords, np.zeros(len(coords))])

        point_labels = ["Query"] + [f"S{i+1}" for i in range(len(sentences))]
        colors = ["#7c8cf8"] + ["#5ee7a0" if s >= 0.65 else "#f5c542" if s >= 0.35 else "#f87171"
                                 for s in scores]
        sizes  = [18] + [10] * len(sentences)

        fig_pca = go.Figure()
        # draw lines from query to each point
        for i in range(1, len(coords)):
            fig_pca.add_trace(go.Scatter(
                x=[coords[0,0], coords[i,0]],
                y=[coords[0,1], coords[i,1]],
                mode="lines",
                line=dict(color="#1e2742", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))
        fig_pca.add_trace(go.Scatter(
            x=coords[:,0], y=coords[:,1],
            mode="markers+text",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="#0d0f14")),
            text=point_labels,
            textposition="top center",
            textfont=dict(size=10, color="#c9cfe8"),
            hovertext=[f"{t}<br>Score: {sc:.4f}" if i > 0 else t
                       for i, (t, sc) in enumerate(zip(point_labels, [1.0] + list(scores)))],
            hoverinfo="text",
            showlegend=False,
        ))
        fig_pca.update_layout(
            **make_dark_layout(),
            height=420,
            xaxis=dict(gridcolor="#1e2330", zeroline=False, title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)"),
            yaxis=dict(gridcolor="#1e2330", zeroline=False, title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)" if n_comp > 1 else "PC2"),
            title=dict(text="Sentences projected to 2-D via PCA — closer = more similar", font=dict(size=13, color="#7c8cf8"), x=0),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # ══════════════════════════════════════════════════
    #  Paul's Critical Thinking Standards
    # ══════════════════════════════════════════════════
    st.markdown("### Paul's Critical Thinking Standards")
    best_sent, best_score = ranked_sentences[0], ranked_scores[0]
    worst_sent, worst_score = ranked_sentences[-1], ranked_scores[-1]

    standards = [
        ("Clarity",
         f'The query was: <em>"{query}"</em>. Each comparison sentence was embedded using a pretrained transformer and ranked by cosine similarity to the query embedding.'),
        ("Accuracy",
         "Model used: <strong>all-MiniLM-L6-v2</strong> from Sentence Transformers (Hugging Face). No claims are made beyond what the model's cosine similarity score directly indicates."),
        ("Precision",
         f"Exact scores range from <strong>{min(scores):.4f}</strong> to <strong>{max(scores):.4f}</strong>. The average is <strong>{np.mean(scores):.4f}</strong>. Scores are cosine values on a 0–1 scale."),
        ("Relevance",
         "All three graphs — bar chart, heatmap, and PCA plot — directly visualise the similarity values produced by the model. No decorative elements are included."),
        ("Logic",
         f'The top result ("<em>{best_sent[:80]}{"…" if len(best_sent)>80 else ""}</em>", score {best_score:.4f}) ranks highest because its semantic embedding is geometrically closest to the query vector in 384-dimensional space.'),
        ("Significance",
         f"The most important finding is that the top-ranked sentence achieves a score of <strong>{best_score:.4f}</strong>, indicating {'strong' if best_score>=0.65 else 'moderate' if best_score>=0.35 else 'weak'} semantic overlap with the query."),
        ("Fairness",
         "Limitation: all-MiniLM-L6-v2 is a general-purpose English model. It may underperform on domain-specific jargon, non-English text, or very short single-word inputs. It also reflects biases present in its training corpus."),
    ]

    cols = st.columns(2)
    for i, (label, text) in enumerate(standards):
        with cols[i % 2]:
            st.markdown(
                f'<div class="ct-box"><div class="ct-label">{label}</div>{text}</div>',
                unsafe_allow_html=True,
            )

else:
    st.info("Enter your sentences above and click **Analyse Similarity** to begin.")
