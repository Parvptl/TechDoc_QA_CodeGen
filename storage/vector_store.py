class VectorStore:
    """Mock/Stub for Vector Store since we are using HybridRetriever directly for zero-setup."""
    def __init__(self):
        pass
        
    def load_index(self):
        return True
