import logging
import os
import torch

from abc import ABCMeta
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder as STCrossEncoder

cross_encoder_dict = {}
os.environ["TOKENIZERS_PARALLELISM"] = "true"

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class CrossEncoderMeta(ABCMeta):
    """Metaclass for keeping track of all 'CrossEncoder' subclasses"""
    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)
        if name != "CrossEncoder":
            cross_encoder_dict[name.lower()] = new_cls
        return new_cls


class CrossEncoder(metaclass=CrossEncoderMeta):
    
    def __init__(self, config=None):
        self.config = config if config is not None else {"type": "minilm_l6_v2", "kwargs": {}}      
          
        cross_encoder_type = self.config.get("type")
        
        self.subclass = cross_encoder_dict[cross_encoder_type]()
    
    def predict(self, *args, **kwargs):
        return self.subclass.predict(*args, **kwargs)

class MiniLM_L6_v2(CrossEncoder):
    
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L6-v2'):
        """
        Initializes the CrossEncoderMP with the specified model.
        
        Args:
            model_name (str): Name/path of the CrossEncoder model to load. Defaults to
                            'cross-encoder/ms-marco-TinyBERT-L-2-v2', a small but efficient model.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = STCrossEncoder(model_name, device=self.device)

    # 4096 / 8192 / 16384 / 32768
    def predict(self, query_alias_pairs, entity_indices, batch_size=32768):
        """
        New method for medical entity linking with alias pooling
        
        Args:
            query_alias_pairs: List of (query, alias) tuples
            entity_indices: List mapping each alias to its entity index
            batch_size: Processing batch size
            
        Returns:
            dict: {entity_idx: max_score} across all aliases
            list: All scores in input order (for debugging)
        """
        # Phase 1: Score all query-alias pairs
        all_scores = []
        for i in range(0, len(query_alias_pairs), batch_size):
            batch = query_alias_pairs[i:i+batch_size]
            with torch.no_grad():
                logits = self.model.predict(batch, apply_softmax=False)
                logits_tensor = torch.tensor(logits)
                probs = torch.sigmoid(logits_tensor)
                all_scores.extend(probs.tolist())
        
        # Phase 2: Max-pool by entity
        entity_scores = {}
        for idx, score in zip(entity_indices, all_scores):
            if idx not in entity_scores or score > entity_scores[idx]:
                entity_scores[idx] = score
                
        return entity_scores, all_scores
    
class BioLinkBERT(CrossEncoder):

    def __init__(self, model_name="michiyasunaga/BioLinkBERT-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _score_pairs(self, pairs_chunk):
        texts = [f"{q} [SEP] {a}" for q, a in pairs_chunk]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            # Use raw logits directly for ranking
            return logits.cpu().tolist()


    def predict(self, query_alias_pairs, entity_indices, batch_size=32768):
        all_scores = []
        for i in range(0, len(query_alias_pairs), batch_size):
            batch = query_alias_pairs[i : i + batch_size]
            batch_scores = self._score_pairs(batch)
            probs = torch.sigmoid(batch_scores)
            all_scores.extend(probs)

        entity_scores = {}
        for idx, score in zip(entity_indices, all_scores):
            if idx not in entity_scores or score > entity_scores[idx]:
                entity_scores[idx] = score

        return entity_scores, all_scores