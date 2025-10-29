import os
import pickle
import joblib

from typing import Any, Dict, Optional, Tuple, Sequence
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize
from xmr4el.models.featurization_wrapper.dimension_model import DimensionModel
from xmr4el.models.featurization_wrapper.transformers import Transformer
from xmr4el.models.featurization_wrapper.vectorizers import Vectorizer


class TextEncoder():
    
    def __init__(
        self,
        vectorizer_config: Optional[Dict[str, Any]] = None,
        transformer_config: Optional[Dict[str, Any]] = None,
        dimension_config: Optional[Dict[str, Any]] = None,
        flag: int = 2,
    ) -> None:
        """Create a new :class:`TextEncoder`."""
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        
        self._vectorizer_model: Optional[Vectorizer] = None
        self._dimension_model: Optional[DimensionModel] = None
        self.flag = flag
    
    @property
    def vectorizer_model(self) -> Optional[Vectorizer]:
        """Return the fitted vectorizer model, if any."""
        return self._vectorizer_model
    
    @vectorizer_model.setter
    def vectorizer_model(self, value: Vectorizer) -> None:
        """Set the fitted vectorizer model."""
        self._vectorizer_model = value
        
    @property
    def dimension_model(self) -> Optional[DimensionModel]:
        """Return the fitted dimensionality reduction model."""
        return self._dimension_model
    
    @dimension_model.setter
    def dimension_model(self, value: DimensionModel) -> None:
        """Set the fitted dimensionality reduction model."""
        self._dimension_model = value
        
    def save(self, save_dir: str) -> None:
        """Persist the encoder and its models to ``save_dir``."""
        os.makedirs(save_dir, exist_ok=True)

        state = self.__dict__.copy()
        models = ["vectorizer_model", "dimension_model"]
        models_data = [getattr(self, model_name, None) for model_name in models]

        for idx, model in enumerate(models_data):
            if model is not None:
                model_path = os.path.join(save_dir, models[idx])
                if hasattr(model, "save"):
                    model.save(model_path)
                else:
                    try:
                        joblib.dump(model, f"{model_path}.joblib")
                    except ImportError:
                        with open(f"{model_path}.pkl", "wb") as f:
                            pickle.dump(model, f)
                state.pop(models[idx], None)

        with open(os.path.join(save_dir, "text_encoder.pkl"), "wb") as fout:
            pickle.dump(state, fout)


    @classmethod
    def load(cls, load_dir: str) -> "TextEncoder":
        """Load a previously saved :class:`TextEncoder` instance."""
        text_encoder_path = os.path.join(load_dir, "text_encoder.pkl")
        assert os.path.exists(text_encoder_path), f"Text Encoder path {text_encoder_path} does not exist"

        with open(text_encoder_path, "rb") as fin:
            model_data = pickle.load(fin)

        model = cls()
        model.__dict__.update(model_data)
        
        # Load models
        model_files = {
            "vectorizer_model": Vectorizer if hasattr(Vectorizer, "load") else None,
            "dimension_model": DimensionModel if hasattr(DimensionModel, "load") else None,
        }
        
        for model_name, model_class in model_files.items():
            model_path = os.path.join(load_dir, model_name)
            if os.path.exists(model_path) and model_class is not None:
                setattr(model, model_name, model_class.load(model_path))
            else:
                if model.flag == 3:
                    setattr(model, model_name, None)
                else:
                    raise Exception("Something with the loading the models is not right")
            
        return model
    
    @staticmethod
    def _reduce_dimensions(
        X_emb: csr_matrix, dim_config: Optional[Dict[str, Any]]
    ) -> Tuple[csr_matrix, DimensionModel]:
        """Reduce dimensionality of sparse embeddings."""
        if dim_config is None:
            print("Running on default config of TruncateSVD")
        
        model = DimensionModel.fit(X_emb, dim_config)
        reduced_emb = model.transform(X_emb)
        return reduced_emb, model
    
    @staticmethod 
    def _predict_dimension(X_emb: csr_matrix, dim_model: DimensionModel) -> csr_matrix:
        """Project embeddings using an existing dimensionality reduction model."""
        if dim_model is None:
            raise AttributeError("No model found in dim_model")
        return dim_model.transform(X_emb)
    
    @staticmethod
    def _encode_text_using_text_vectorizer(
        X_test: Sequence[str], vec_config: Optional[Dict[str, Any]]
    ) -> Tuple[csr_matrix, Vectorizer]:
        """Fit a text vectorizer and transform input texts."""
        if vec_config is None:
            print("Running on default config of TF-IDF")
            
        model = Vectorizer.fit(X_test, vec_config)
        sparse_emb = model.transform(X_test) # CSR_MATRIX    
        return sparse_emb, model
    
    @staticmethod
    def _predict_text_using_text_vectorizer(
        X_test: Sequence[str], vec_model: Vectorizer
    ) -> csr_matrix:
        """Transform texts using an existing vectorizer."""
        if vec_model is None:
            raise AttributeError("No model found in vec_model")
        return vec_model.transform(X_test)
    
    @staticmethod
    def _encode_text_using_transformer(
        X_test: Sequence[str], transformer_config: Optional[Dict[str, Any]]
    ) -> Any:
        """Encode texts using a Transformer model."""
        if transformer_config is None:
            print("Running on default config of BioBert")
        
        _, transformer_embeddings = Transformer.transform(X_test, transformer_config)
        return transformer_embeddings
    
    @staticmethod
    def _predict_text_using_transformer(
        X_test: Sequence[str], transformer_config: Dict[str, Any]
    ) -> Any:
        """Predict embeddings for ``X_test`` using a Transformer model."""
        if transformer_config is None:
            raise AttributeError("No config found in transformer_config")
        _, transformer_embeddings = Transformer.transform(X_test, transformer_config)
        return transformer_embeddings
        
    def encode(self, X_test: Sequence[str]) -> csr_matrix:
        """Encode input texts into normalized feature vectors."""
        use_tfidf = self.flag in [1, 2]
        use_transformer = self.flag in [2, 3]

        if use_tfidf:
            X_tfidf, vec_model = self._encode_text_using_text_vectorizer(
                X_test, self.vectorizer_config
            )
            reduced_x_tfidf, dim_model = self._reduce_dimensions(
                X_tfidf, self.dimension_config
            )

            if dim_model is None:
                reduced_x_tfidf = X_tfidf
            else:
                reduced_x_tfidf = csr_matrix(reduced_x_tfidf)

            self.vectorizer_model = vec_model
            self.dimension_model = dim_model

        if use_transformer:
            X_transformer = self._encode_text_using_transformer(
                X_test, self.transformer_config
            )
            sparse_X_transformer = csr_matrix(X_transformer)

            if use_tfidf:
                concat_emb = hstack([reduced_x_tfidf, sparse_X_transformer])
            else:
                concat_emb = sparse_X_transformer
        else:
            concat_emb = reduced_x_tfidf

        concat_emb = normalize(concat_emb, norm="l2", axis=1)
        return concat_emb
    
    def predict(self, X_text_query: Sequence[str]) -> csr_matrix:
        """Encode query data for prediction."""
        use_tfidf = self.flag in [1, 2]
        use_transformer = self.flag in [2, 3]

        if use_tfidf:
            X_tfidf_query = self._predict_text_using_text_vectorizer(
                X_test=X_text_query, vec_model=self.vectorizer_model
            )

            if self.dimension_model is None:
                reduced_x_tfidf = X_tfidf_query
            else:
                reduced_x_tfidf = csr_matrix(
                    self._predict_dimension(X_tfidf_query, self.dimension_model)
                )

        if use_transformer:
            X_transformer = self._predict_text_using_transformer(
                X_test=X_text_query, transformer_config=self.transformer_config
            )
            sparse_X_transformer = csr_matrix(X_transformer)

            if use_tfidf:
                concat_emb = hstack([reduced_x_tfidf, sparse_X_transformer])
            else:
                concat_emb = sparse_X_transformer
        else:
            concat_emb = reduced_x_tfidf

        concat_emb = normalize(concat_emb, norm="l2", axis=1)
        return concat_emb