# NLP Functions

# Function to remove geographic info from descriptions
def remove_geo_entities(text):
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