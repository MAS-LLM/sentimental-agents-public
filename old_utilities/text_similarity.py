from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class SentenceEmbedder:
    """
    A class to embed sentences using specified transformer models.

    Attributes:
        tokenizer: Tokenizer for the specified model.
        model: The pretrained transformer model for embedding generation.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the SentenceEmbedder with the given model name.
        
        Args:
            model_name (str): Name of the model to be loaded from HuggingFace's model hub.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies mean pooling on the model output to get sentence embeddings.
        
        Args:
            model_output (torch.Tensor): Output from the transformer model.
            attention_mask (torch.Tensor): Mask to avoid counting padding tokens.
            
        Returns:
            torch.Tensor: Pooled sentence embeddings.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: [str]) -> torch.Tensor:
        """
        Encodes a list of texts into embeddings.
        
        Args:
            texts ([str]): List of sentences to encode.
            
        Returns:
            torch.Tensor: Sentence embeddings.
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)
    
    def is_match(self, query: str, doc: str, threshold: float = 0.75) -> bool:
        """
        Determines if a query matches a document based on their similarity score.
        
        Args:
            query (str): Query text.
            doc (str): Document text.
            threshold (float): Similarity score threshold to consider as a match.
        
        Returns:
            bool: True if there's a match, False otherwise.
        """
        # Compute embeddings for both the query and the document
        query_emb = self.encode([query])
        doc_emb = self.encode([doc])

        # Calculate similarity score
        score = torch.mm(query_emb, doc_emb.transpose(0, 1))[0][0].item()
        return score > threshold

def get_high_score_texts(query_emb: torch.Tensor, doc_emb: torch.Tensor, docs: [str], threshold: float = 0.75) -> [str]:
    """
    Returns texts that have a similarity score above the given threshold.
    
    Args:
        query_emb (torch.Tensor): Embedding of the query text.
        doc_emb (torch.Tensor): Embeddings of the documents.
        docs ([str]): List of original document texts.
        threshold (float): Similarity score threshold.
        
    Returns:
        [str]: Documents with similarity scores above the threshold.
    """
    scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()
    doc_score_pairs = list(zip(docs, scores))
    
    # Filter the documents by score threshold
    high_score_texts = [doc for doc, score in doc_score_pairs if score > threshold]
    
    return high_score_texts




if __name__ == "__main__":
    query = "Safety procedures are crucial to ensuring the safety of the candidate"
    docs = ["We must absolutely probe the candidate's knowledge of safety procedures during the interview!", 
            "The candidate's CV shouldn't definitely highlight their expertise in safety procedures!"]
    
    embedder = SentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    query_emb = embedder.encode(query)
    doc_emb = embedder.encode(docs)
    
    relevant_docs = get_high_score_texts(query_emb, doc_emb, docs)
    for doc in relevant_docs:
        print(doc)


    embedder = SentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    doc = "We must absolutely probe the candidate's knowledge of safety procedures during the interview!"
    
    if embedder.is_match(query, doc):
        print("Match found!")
    else:
        print("No match found.")
