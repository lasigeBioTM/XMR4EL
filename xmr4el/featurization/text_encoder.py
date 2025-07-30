import os
import pickle
import joblib

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import normalize

from xmr4el.models.featurization_wrapper.dimension_model import DimensionModel
from xmr4el.models.featurization_wrapper.transformers import Transformer
from xmr4el.models.featurization_wrapper.vectorizers import Vectorizer


class TextEncoder():
    
    def __init__(self, 
                 vectorizer_config=None, 
                 transformer_config=None, 
                 dimension_config=None,
                 flag = 2 # 1 -> Use tf-idf, 2 -> use both
                 ):
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        
        self._vectorizer_model = None
        self._dimension_model = None
        self.flag = flag
    
    @property
    def vectorizer_model(self):
        return self._vectorizer_model
    
    @vectorizer_model.setter
    def vectorizer_model(self, value):
        self._vectorizer_model = value
        
    @property
    def dimension_model(self):
        return self._dimension_model
    
    @dimension_model.setter
    def dimension_model(self, value):
        self._dimension_model = value
        
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    
        state = self.__dict__.copy()
        
        models = ["vectorizer_model", "dimension_model"] # reranker
        models_data = [getattr(self, model_name, None) for model_name in models]

        for idx, model in enumerate(models_data):
            if model is not None:  # Ensure model exists before saving
                model_path = os.path.join(save_dir, models[idx])
                # Handle models with different save methods
                if hasattr(model, 'save'):
                    model.save(model_path)
                else:
                    # For models without save method, use joblib or pickle
                    try:
                        joblib.dump(model, f"{model_path}.joblib")
                    except ImportError:
                        with open(f"{model_path}.pkl", "wb") as f:
                            pickle.dump(model, f)
                            
                state.pop(model, None) # Delete the model
        
        with open(os.path.join(save_dir, "text_encoder.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir):
        text_encoder_path = os.path.join(load_dir, "text_encoder.pkl")
        assert os.path.exists(text_encoder_path), f"Text Encoder path {text_encoder_path} does not exist"

        with open(text_encoder_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        # Load models
        model_files = {
            "vectorizer_model": Vectorizer if hasattr(Vectorizer, 'load') else None,
            "dimension_model": DimensionModel if hasattr(DimensionModel, 'load') else None,
        }
        
        for model_name, model_class in model_files.items():
            model_path = os.path.join(load_dir, model_name)
            # First check for model-specific save format
            if os.path.exists(model_path) and model_class is not None:
                setattr(model, model_name, model_class.load(model_path))
            else:
                raise "Some Model is does not have an load() method"
            
        return model
    
    @staticmethod
    def _reduce_dimensions(X_emb, dim_config):
        if dim_config is None:
            print("Running on default config of TruncateSVD")
        
        model = DimensionModel.fit(X_emb, dim_config)
        reduced_emb = model.transform(X_emb)
        return reduced_emb, model
    
    @staticmethod
    def _encode_text_using_text_vectorizer(X_test, vec_config):
        if vec_config is None:
            print("Running on default config of TF-IDF")
            
        model = Vectorizer.fit(X_test, vec_config)
        sparse_emb = model.transform(X_test) # CSR_MATRIX    
        return sparse_emb, model
    
    @staticmethod
    def _encode_text_using_transformer(X_test, transformer_config):
        if transformer_config is None:
            print("Running on default config of BioBert")
        
        model = Transformer.transform(X_test, transformer_config)
        dense_emb = model.embeddings
        return dense_emb, model
        
    def encode(self, X_test):
        """Encode the training data"""
        
        X_tfidf, vec_model = self._encode_text_using_text_vectorizer(X_test, self.vectorizer_config) # sparse
        
        reduced_x_tfidf, dim_model = self._reduce_dimensions(X_tfidf, self.dimension_config) # dense
        if dim_model is None:
            reduced_x_tfidf = X_tfidf # sparse
        else:
            reduced_x_tfidf = csr_matrix(reduced_x_tfidf) # sparse
        
        if self.flag == 2:
            X_transformer, transformer_model = self._encode_text_using_transformer(X_test, self.transformer_config) # dense
            sparse_X_transformer = csr_matrix(X_transformer) # sparse
            concat_emb = hstack([reduced_x_tfidf, sparse_X_transformer])
            
        else:
            concat_emb = reduced_x_tfidf
        
        concat_emb = normalize(concat_emb, norm="l2", axis=1)

        self.vectorizer_model = vec_model
        self.dimension_model = dim_model

        return concat_emb
    
            

        

            
        
        
    