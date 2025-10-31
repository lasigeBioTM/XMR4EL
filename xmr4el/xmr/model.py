import os
import joblib
import pickle
import time 
import warnings
import logging

import numpy as np

from xmr4el import get_logger, set_verbosity
from numpy import asarray, int32, argpartition, argsort, float32
from datetime import datetime
from typing import Optional
from memory_profiler import profile
from scipy.sparse import csr_matrix
from xmr4el.featurization.label_embedding_factory import LabelEmbeddingFactory
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.featurization.text_encoder import TextEncoder
from xmr4el.xmr.base import HierarchicaMLModel
from xmr4el.utils.temp_store import TempVarStore


os.makedirs("/tmp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


class XModel():
    
    def __init__(self, 
                 vectorizer_config: dict = None,
                 transformer_config: dict = None,
                 dimension_config: dict = None,
                 clustering_config: dict = None,
                 matcher_config: dict = None,
                 ranker_config: dict = None,
                 cur_config: dict = None,
                 min_leaf_size: int = 20,
                 max_leaf_size: int = None,
                 cut_half_cluster: bool = False,
                 ranker_every_layer: bool = True,
                 n_workers: int = 8,
                 depth: int = 1,
                 emb_flag: int = 1,
                 verbose: Optional[int] = None,
                 logger: Optional[logging.Logger] = None
                 ):
        
        # 0 = WARNING, 1 = INFO, 2 = DEBUG
        if logger is not None:
            self.logger = logger
            self.logger.debug("Using user-supplied logger for XModel")
        else:
            if verbose is not None:
                set_verbosity(verbose)
            self.logger = get_logger("models.xmodel")

        # rest of initialization
        self.logger.info("Initializing XModel")
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.ranker_config = ranker_config
        self.cur_config = cur_config
        
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.cut_half_cluster = cut_half_cluster
        self.ranker_every_layer = ranker_every_layer
        
        self.n_workers = n_workers
        
        self.depth = depth
        self.emb_flag =emb_flag
        
        self._text_encoder = None
        self._hml = None
        self._training_texts = None
        self._original_labels = None
        self._X = None
        self._Y = None
        self._Z = None
    
        self.EPS = 1e-12
        self.TOPK_DBG = 50
        
        self.temp_var = TempVarStore()
    
    def __str__(self) -> str:
        """Human-friendly multi-line summary of the model and its config/state."""
        def _short(obj, max_len=100):
            """Return a short representation for objects (dicts/lists/others)."""
            try:
                if obj is None:
                    return "None"
                
                if isinstance(obj, dict):
                    # show keys and count
                    keys = list(obj.keys())
                    k_display = ", ".join(map(str, keys[:6]))
                    more = f", ... (+{len(keys)-6})" if len(keys) > 6 else ""
                    return f"dict(keys=[{k_display}{more}])"
                    
                if isinstance(obj, (list, tuple, set)):
                    n = len(obj)
                    # preview = ", ".join(repr(x) for x in list(obj)[:6])
                    # more = f", ... (+{n-6})" if n > 6 else ""
                    return f"{type(obj).__name__}(len={n})"
                    
                # for numpy arrays / pandas objects show shape/len if possible
                if hasattr(obj, "shape"):
                    return f"{type(obj).__name__}(shape={getattr(obj, 'shape')})"
                
                if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes)):
                    return f"{type(obj).__name__}(len={len(obj)})"
                
                r = repr(obj)
                return r if len(r) <= max_len else r[:max_len] + "..."
            
            except Exception:
                return f"<unrepr {type(obj).__name__}>"

        logger_name = getattr(self, "logger", None)
        if logger_name is not None:
            try:
                logger_info = f"{self.logger.name} (level={self.logger.level})"
            except Exception:
                logger_info = repr(self.logger)
        else:
            logger_info = "None"

        parts = [
            f"XModel summary:",
            f"  logger: {logger_info}",
            f"  workers: n_workers={self.n_workers}",
            f"  depth={self.depth}, emb_flag={self.emb_flag}",
            f"  cluster: min_leaf_size={self.min_leaf_size}, max_leaf_size={self.max_leaf_size}, cut_half_cluster={self.cut_half_cluster}",
            f"  ranker_every_layer={self.ranker_every_layer}",
            f"  configs:",
            f"    vectorizer: {_short(self.vectorizer_config)}",
            f"    transformer: {_short(self.transformer_config)}",
            f"    dimension: {_short(self.dimension_config)}",
            f"    clustering: {_short(self.clustering_config)}",
            f"    matcher: {_short(self.matcher_config)}",
            f"    ranker: {_short(self.ranker_config)}",
            f"    cur: {_short(self.cur_config)}",
            f"  internal state:",
            f"    text_encoder: {_short(self._text_encoder)}",
            f"    hml: {_short(self._hml)}",
            f"    training_texts: {_short(self._training_texts)}",
            f"    original_labels: {_short(self._original_labels)}",
            f"    X/Y/Z: {_short(self._X)}, {_short(self._Y)}, {_short(self._Z)}",
            f"  temp_var: {_short(getattr(self, 'temp_var', None))}",
        ]
        return "\n".join(parts)

    def __repr__(self) -> str:
        # concise repr that can be used in containers / REPL
        try:
            return f"XModel(depth={self.depth}, n_workers={self.n_workers}, emb_flag={self.emb_flag})"
        except Exception:
            return "<XModel (repr error)>"

    
    @property
    def text_encoder(self):
        return self._text_encoder
    
    @text_encoder.setter
    def text_encoder(self, value):
        self._text_encoder = value

    @property
    def model(self):
        return self._hml
    
    @model.setter
    def model(self, value):
        self._hml = value
        
    @property
    def training_set(self):
        return self._training_texts
    
    @training_set.setter
    def training_set(self, value):
        self._training_texts = value
        
    @property
    def initial_labels(self):
        return self._original_labels
    
    @initial_labels.setter
    def initial_labels(self, value):
        self._original_labels = value
        
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, value):
        self._X = value
        
    @property
    def Y(self):
        return self._Y
    
    @Y.setter
    def Y(self, value):
        self._Y = value
        
    @property 
    def Z(self):
        return self._Z
    
    @Z.setter
    def Z(self, value):
        self._Z = value        
        
    def save(self, save_dir):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(save_dir, f"{self.__class__.__name__.lower()}_{timestamp}")
        os.makedirs(save_dir, exist_ok=False)
    
        state = self.__dict__.copy()
        
        model = self.model
        model_path = os.path.join(save_dir, "hml")
        
        if model is not None:
            if hasattr(model, "save"):
                model.save(model_path)
            else:
                try:
                    joblib.dump(model, f"{model_path}.joblib")
                except ImportError:
                    with open(f"{model_path}.pkl", "wb") as f:
                        pickle.dump(model, f)  
            
            state.pop("_hml", None) # Popped _hml from class
        
        text_encoder = self.text_encoder
        text_encoder_path = os.path.join(save_dir, "text_encoder")
        
        if text_encoder is not None:
            if hasattr(text_encoder, "save"):
                text_encoder.save(text_encoder_path)
            else:
                try:
                    joblib.dump(text_encoder, f"{text_encoder_path}.joblib")
                except ImportError:
                    with open(f"{text_encoder_path}.pkl", "wb") as f:
                        pickle.dump(text_encoder, f)  
                          
        with open(os.path.join(save_dir, "xmodel.pkl"), "wb") as fout:
            pickle.dump(state, fout)
    
    @classmethod
    def load(cls, load_dir):
        xmodel_path = os.path.join(load_dir, "xmodel.pkl")
        assert os.path.exists(xmodel_path), f"XModel path {xmodel_path} does not exist"
        
        with open(xmodel_path, "rb") as fin:
            model_data = pickle.load(fin)
            
        model = cls()
        model.__dict__.update(model_data)
        
        model_path = os.path.join(load_dir, "hml")
        hml = HierarchicaMLModel.load(model_path)
        setattr(model, "_hml", hml)
        
        text_encoder_path = os.path.join(load_dir, "text_encoder")
        text_encoder = TextEncoder.load(text_encoder_path)
        setattr(model, "_text_encoder", text_encoder)
        
        return model
    
    def _fit(self, X_text, Y_text):
        """Returns embeddings: ndarray"""
        
        self.initial_labels = self.temp_var.save_model_temp(Y_text)
        self.training_set = self.temp_var.save_model_temp(X_text)
        
        self.logger.info("Preparing Data")
        
        X_processed, Y_label_to_indices = Preprocessor.prepare_data(X_text, Y_text)
        
        self.logger.info("Started Encoding")
        
        # Encode X_processed
        text_encoder = TextEncoder(
            vectorizer_config=self.vectorizer_config,
            transformer_config=self.transformer_config,
            dimension_config=self.dimension_config, 
            flag=self.emb_flag # Needs to be a variable, could have a stop to check
            )
        
        self.text_encoder = text_encoder
        
        X_emb = text_encoder.encode(X_processed)
        
        Y_label_matrix = LabelEmbeddingFactory.generate_label_matrix(Y_label_to_indices)
        
        # Process Labels
        Y_binazer, _ = LabelEmbeddingFactory.label_binarizer(Y_label_matrix)
        Z = LabelEmbeddingFactory.generate_PIFA(X_emb, Y_binazer)
        
        return X_emb, Y_binazer, Z 
    
    # @profile
    def train(self, X_text, Y_text):
        
        self.logger.info("Started Training")
        
        self.X, self.Y, self.Z = self._fit(X_text=X_text, Y_text=Y_text)

        n_labels = self.Z.shape[0]
        local_to_global = np.arange(n_labels, dtype=int)
        global_to_local = {g: i for i, g in enumerate(local_to_global)}

        self.logger.info("Hierarchical Model Pipeline")

        hml = HierarchicaMLModel(
            clustering_config=self.clustering_config,
            matcher_config=self.matcher_config,
            ranker_config=self.ranker_config,
            cur_config=self.cur_config,
            min_leaf_size=self.min_leaf_size,
            max_leaf_size=self.max_leaf_size,
            n_workers=self.n_workers,
            cut_half_cluster=self.cut_half_cluster,
            ranker_every_layer=self.ranker_every_layer,
            layer=self.depth
        )

        hml.train(
            X_train=self.X,
            Y_train=self.Y,
            Z_train=self.Z,
            local_to_global=local_to_global,
            global_to_local=global_to_local
        )

        self.model = hml
        
        self.initial_labels = self.temp_var.load_model_temp(self.initial_labels)
        self.training_set = self.temp_var.load_model_temp(self.training_set)
        
        self.temp_var.delete_model_temp()
        
    def predict(self, X_text, 
                topk: int = 5, 
                beam_size: int | None = None, 
                fusion: str = "geometric", 
                topk_mode: str = "per_leaf", 
                topk_inside_global: int | None = None,
                n_jobs: int =-1):
            """Predict label scores for given text inputs.

            Parameters
            ----------
            X_text : list-like
                Raw text queries.
            topk : int, optional
                Number of labels to consider when computing hit counts.
            beam_size : int, optional
                Beam width for hierarchical traversal.
            golden_labels : Sequence[Sequence[str]], optional
                Gold standard label IDs per query (strings).
            return_hits : bool, optional
                If ``True`` and ``golden_labels`` provided, returns hit counts.

            Returns
            -------
            csr_matrix
                Sparse score matrix for all queries.
            list, optional
                Hit counts per query when ``return_hits`` is ``True``.
            """
            
            time_start_encoding = time.time()

            X_query = self.text_encoder.predict(X_text)
            
            time_end_encoding = time.time()
            
            print("Encoding: ", time_end_encoding - time_start_encoding)
            
            if topk_mode == "per_leaf":
                return self.model.predict(X_query, 
                                          topk=topk, 
                                          beam_size=beam_size, 
                                          fusion=fusion, 
                                          n_jobs=n_jobs, 
                                          topk_mode=topk_mode)
            """
                Nice touch, it makes it evaluate all the leaf labels, could be problematic if beam size to great, 
                Maybe prune anyway ? could be a option
            """   
            out_h, _ = self.model.predict(
                X_query,
                topk=topk_inside_global,
                beam_size=beam_size,
                fusion=fusion,
                n_jobs=-1,
                topk_mode="per_leaf",  # do not truncate; gather full union from leaves
            )
            
            time_start_reranking = time.time()
            
            n_queries = X_query.shape[0]
            
            # 3) prepare label embedding bank Z (must be same space as X_query) and L2-normalize once
            if getattr(self, "_Z", None) is None:
                raise RuntimeError("label_embeddings (Z) not found on hierarchical model.")
            Z = self.Z
            n_labels, D = Z.shape


            Zf = Z.astype(np.float32, copy=False)
            norms = np.linalg.norm(Zf, axis=1, keepdims=True) + 1e-12
            Z_norm = Zf / norms  # (n_labels, D)

            # 4) collect union of candidate label IDs per query from leaf paths
            cand_ids_per_q = []
            for qi in range(n_queries):
                lids_set = set()
                for p in out_h[qi].get("paths", []):
                    for lid in p.get("leaf_global_labels", []):
                        lid_int = int(lid)
                        if 0 <= lid_int < n_labels:
                            lids_set.add(lid_int)
                cand_ids_per_q.append(sorted(lids_set))

            # If no candidates at all, return empty CSR with same 'out'
            if all(len(lids) == 0 for lids in cand_ids_per_q):
                return out_h, csr_matrix((n_queries, n_labels), dtype=np.float32)

            # build cosine scores per query over the candidate subset and assemble CSR
            indptr = [0]
            indices = []
            data = []

            use_topk = (topk is not None and topk > 0)

            def _row_to_dense_norm(xrow):
                if hasattr(xrow, "toarray"):  # sparse
                    v = xrow.toarray().ravel().astype(np.float32, copy=False)
                else:
                    v = np.asarray(xrow, dtype=np.float32).ravel()
                n = np.linalg.norm(v) + self.EPS
                return v / n

            for qi in range(n_queries):
                cands = cand_ids_per_q[qi]
                if not cands:
                    indptr.append(len(indices))
                    continue

                qv = _row_to_dense_norm(X_query[qi])
                cands_arr = asarray(cands, dtype=int32)
                Lsub = Z_norm[asarray(cands, dtype=int32)]  # (K, D), already L2-normed
                sims = Lsub @ qv                                  # (K,), cosine = dot

                # global top-k (per query)
                if use_topk and sims.size > topk:
                    idx = argpartition(sims, sims.size - topk)[-topk:]
                    ord_ = argsort(-sims[idx])
                    sel = idx[ord_]
                else:
                    sel = argsort(-sims)

                chosen_ids = cands_arr[sel]
                chosen_scores = sims[sel].astype(float32, copy=False)

                indices.extend(chosen_ids.tolist())
                data.extend(chosen_scores.tolist())
                indptr.append(len(indices))

            scores_cos = csr_matrix(
                (np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int32)),
                shape=(n_queries, n_labels),
                dtype=np.float32
            )
            
            time_end_reranking = time.time()
            
            print("Reranking: ", time_end_reranking - time_start_reranking)

            # 6) IMPORTANT: return the SAME 'out' from HMLModel, only scores are replaced by cosine
            return out_h, scores_cos
    