# NLP Functions
import pandas as pd
import numpy as np
import html 
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import torch

# ==============================================================================
# CLEAN DATA 
# ==============================================================================
def clean_description_text(df, text_col="description"):
    """
    Clean text descriptions by:
    - filling missing values
    - converting to string
    - decoding HTML entities
    - removing extra whitespace
    - trimming leading/trailing spaces

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    text_col : str
        Name of text column to clean.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned text column.
    """

    df = df.copy()

    df[text_col] = (
        df[text_col]
        .fillna("")
        .astype(str)
        .apply(html.unescape)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


# Function to remove geographic info from descriptions
def remove_geo_entities(text,nlp):
    doc = nlp(text)
    cleaned = text

    for ent in reversed(doc.ents):
        if ent.label_ in ["GPE", "LOC", "NORP"]:
            cleaned = (
                cleaned[:ent.start_char]
                + " "
                + cleaned[ent.end_char:]
            )
    return " ".join(cleaned.split())

# ==============================================================================
# BERT
# ==============================================================================
# Override the MPS deserializer to map to CPU in non-Mac environments
def _mps_override(obj, location):
    """
    Deserialization monkey-patch for PyTorch model loading across cross-platform environments.

    Redirects PyTorch storage tensors serialized on Apple Silicon (MPS/Metal Performance Shaders) 
    to system CPU memory. This resolves runtime errors when loading Mac-saved models inside 
    non-macOS environments (e.g., Linux containers or GitHub Codespaces).

    Parameters
    ----------
    obj : torch.UntypedStorage
        The raw uninitialized storage object managed by PyTorch's legacy deserializer.
    location : str
        The target hardware location specified in the pickled file (e.g., "mps:0").

    Returns
    -------
    torch.UntypedStorage or None
        A CPU-mapped storage object if `location` starts with "mps", otherwise None 
        to pass handling back to the remaining PyTorch package registry functions.
    """
    if location.startswith("mps"):
        return obj.cpu()
    return None

def generate_embeddings(
    documents,
    model_name="all-MiniLM-L6-v2",
    show_progress_bar=True
):
    """
    Generate sentence embeddings from text documents.

    Parameters
    ----------
    documents : list-like
        Collection of text documents.

    model_name : str
        SentenceTransformer model name.

    show_progress_bar : bool
        Whether to display encoding progress.

    Returns
    -------
    tuple
        (model, embeddings)
    """

    model = SentenceTransformer(
        model_name
    )

    embeddings = model.encode(
        list(documents),
        show_progress_bar=show_progress_bar
    )

    return model, embeddings

def build_vectorizer(
    stop_words=None,
    min_df=5,
    ngram_range=(1, 3)
):
    """
    Create CountVectorizer for BERTopic.

    Parameters
    ----------
    stop_words : list, optional
        Custom stopword list.

    min_df : int
        Minimum document frequency.

    ngram_range : tuple
        N-gram range.

    Returns
    -------
    CountVectorizer
    """

    return CountVectorizer(
        stop_words=stop_words,
        min_df=min_df,
        ngram_range=ngram_range
    )

def build_topic_model(
    embedding_model,
    vectorizer_model,
    random_state=42,
    min_topic_size=20
):
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state
    )

    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=True
    )

    return topic_model

def fit_and_reduce_topics(
    topic_model,
    documents,
    embeddings,
    nr_topics=10
):
    topics, probs = topic_model.fit_transform(
        documents,
        embeddings
    )

    topic_model = topic_model.reduce_topics(
        documents,
        nr_topics=nr_topics
    )

    return topic_model, topics, probs

def add_topic_assignments(
    df,
    topic_model,
    probs
):
    df = df.copy()

    df["topic"] = topic_model.topics_
    df["topic_prob"] = probs.max(axis=1)

    return df
