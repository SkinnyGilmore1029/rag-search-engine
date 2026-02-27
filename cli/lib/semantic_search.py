from sentence_transformers import SentenceTransformer



class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        
def verify_model():
    new_model = SemanticSearch()
    print(f"Model loaded: {new_model.model}")
    print(f"Max sequence length: {new_model.model.max_seq_length}")
