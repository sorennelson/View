

def format_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1000:
        return f"{n/1000:.0f}K"
    else:
        return str(n)

def get_from_chroma_with_ids(collection, ids):
    data = collection.get(ids=ids, include=["documents", "embeddings"])
    return data

def format_impact(distance):
  if distance < 1.2:
    return 'High'
  elif distance < 1.4:
    return 'Medium'
  return 'Low'