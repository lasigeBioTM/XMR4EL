import os
import gc

os.makedirs("/app/joblib_tmp", exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = "/app/joblib_tmp"

import joblib
import pickle
import heapq

import numpy as np

from scipy.sparse import csr_matrix
from scipy.special import expit
from joblib import Parallel, delayed

from datetime import datetime

from xmr4el.featurization.label_embedding_factory import LabelEmbeddingFactory
from xmr4el.featurization.preprocessor import Preprocessor
from xmr4el.featurization.text_encoder import TextEncoder
from xmr4el.xmr.base import HierarchicaMLModel


class XModel():
    
    def __init__(self, 
                 vectorizer_config=None,
                 transformer_config=None,
                 dimension_config=None,
                 clustering_config=None,
                 matcher_config=None,
                 reranker_config=None,
                 min_leaf_size=20,
                 max_leaf_size=None,
                 cut_half_cluster=False,
                 n_workers=8,
                 depth=1,
                 emb_flag=1,
                 ):
        
        self.vectorizer_config = vectorizer_config
        self.transformer_config = transformer_config
        self.dimension_config = dimension_config
        self.clustering_config = clustering_config
        self.matcher_config = matcher_config
        self.reranker_config = reranker_config
        
        self.min_leaf_size = min_leaf_size
        self.max_leaf_size = max_leaf_size
        self.cut_half_cluster = cut_half_cluster
        
        self.n_workers = n_workers
        
        self.depth = depth
        self.emb_flag =emb_flag
        
        self._text_encoder = None
        self._hml = None
        self._original_labels = None
        self._X = None
        self._Y = None
        self._Z = None
    
    
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
        model_path = os.path.join(save_dir, f"hml")
        
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
        text_encoder_path = os.path.join(save_dir, f"text_encoder")
        
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
        
        self.initial_labels = Y_text
        
        X_processed, Y_label_to_indices = Preprocessor.prepare_data(X_text, Y_text)
        
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
    
    def train(self, X_text, Y_text):
        self.X, self.Y, self.Z = self._fit(X_text=X_text, Y_text=Y_text)

        n_labels = self.Z.shape[0]
        local_to_global = np.arange(n_labels, dtype=int)
        global_to_local = {g: i for i, g in enumerate(local_to_global)}

        hml = HierarchicaMLModel(
            clustering_config=self.clustering_config,
            matcher_config=self.matcher_config,
            reranker_config=self.reranker_config,
            min_leaf_size=self.min_leaf_size,
            max_leaf_size=self.max_leaf_size,
            n_workers=self.n_workers,
            cut_half_cluster=self.cut_half_cluster,
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


    @staticmethod
    def _batch_rerank_scores(model_dict, feat_mat):
        if not model_dict:
            return {}

        first_model = next(iter(model_dict.values()))
        if first_model.is_linear_model():
            labels = list(model_dict.keys())
            coef_mat = np.vstack([model_dict[l].coef().ravel() for l in labels])
            intercept_vec = np.array([model_dict[l].intercept()[0] for l in labels])
            raw_scores = feat_mat @ coef_mat.T + intercept_vec
            probs = expit(raw_scores)
            return {lbl: probs[:, i] for i, lbl in enumerate(labels)}
        else:
            # Non-linear: maybe also parallelize here
            return {lbl: model_dict[lbl].predict_proba(feat_mat)[:, 1]
                    for lbl in model_dict}


    def _predict(i, X_query, z_layer, model, initial_labels,
             gold_labels, topk, alpha, beam_size, n_layers):
        
        print(i)
        label_scores = {}
        gold_set = set(gold_labels[i]) if gold_labels else set()

        # Start beam with first ML model at root layer
        beam = [(0, model.hmodel[0][0], 0.0, X_query[i])]
        matched_cluster_found = False

        while beam:
            next_beam = []

            # Group beam items by (layer_idx, ml) to batch predict matcher scores
            beam_groups = {}
            for (layer_idx, ml, cum_score, x_aug) in beam:
                beam_groups.setdefault((layer_idx, ml), []).append((cum_score, x_aug))

            for (layer_idx, ml), entries in beam_groups.items():
                cum_scores, x_augs = zip(*entries)
                x_augs_mat = np.vstack(x_augs)

                # Batch matcher prediction for all x_augs in this group
                matcher_scores_mat = ml.matcher_model.predict_proba(x_augs_mat)  # shape: (batch_size, n_clusters)
                if matcher_scores_mat.size == 0:
                    continue

                z_aug = z_layer[layer_idx]

                for row_idx, matcher_scores in enumerate(matcher_scores_mat):
                    cum_score = cum_scores[row_idx]
                    x_aug = x_augs[row_idx]

                    # Get top-k clusters via argpartition for efficiency
                    if matcher_scores.shape[0] > topk:
                        top_clusters_idx = np.argpartition(matcher_scores, -topk)[-topk:]
                        top_clusters = top_clusters_idx[np.argsort(matcher_scores[top_clusters_idx])[::-1]]
                    else:
                        top_clusters = np.argsort(matcher_scores)[::-1]

                    # Check if any cluster contains gold label
                    for c in top_clusters:
                        cluster_labels = ml.cluster_model.cluster_to_labels[c]
                        if gold_set & {initial_labels[g] for g in cluster_labels}:
                            matched_cluster_found = True
                            break

                    # Prepare batch features for reranker
                    all_global_labels = []
                    all_feat_rows = []
                    label_to_idx = {}

                    total_labels = sum(len(ml.cluster_model.cluster_to_labels[c]) for c in top_clusters)
                    feat_dim = x_aug.shape[0] + z_aug.shape[1]
                    all_feat_rows = np.empty((total_labels, feat_dim), dtype=np.float32)

                    idx = 0
                    for c in top_clusters:
                        cluster_score = matcher_scores[c]
                        global_labels = ml.cluster_model.cluster_to_labels[c]
                        for g in global_labels:
                            all_feat_rows[idx, :x_aug.shape[0]] = x_aug
                            all_feat_rows[idx, x_aug.shape[0]:] = z_aug[g]
                            label_to_idx[g] = idx
                            all_global_labels.append((g, cluster_score))
                            idx += 1

                    rerank_scores = np.array([score for _, score in all_global_labels], dtype=np.float32)

                    # Batch reranker scores
                    if ml.reranker_model is not None and total_labels > 0:
                        rerank_dict = XModel._batch_rerank_scores(ml.reranker_model.model_dict, all_feat_rows)
                        for g, idx_local in label_to_idx.items():
                            if g in rerank_dict:
                                rerank_scores[idx_local] = rerank_dict[g][idx_local]

                    # Fusion and beam expansion
                    for idx_local, (g, cluster_score) in enumerate(all_global_labels):
                        rerank_score = rerank_scores[idx_local]
                        fused = ((cluster_score + 1e-8) ** (1 - alpha)) * ((rerank_score + 1e-8) ** alpha)

                        if fused > label_scores.get(g, -np.inf):
                            label_scores[g] = fused

                        if layer_idx + 1 < n_layers:
                            feat_node = np.array([fused, cluster_score, rerank_score])
                            x_next = np.hstack([x_aug, feat_node])
                            for child_ml in model.hmodel[layer_idx + 1]:
                                if g in child_ml.local_to_global_idx:
                                    next_beam.append((layer_idx + 1, child_ml, cum_score + fused, x_next))

            # Keep top beam_size items for next iteration
            beam = heapq.nlargest(beam_size, next_beam, key=lambda tup: tup[2])

        # Gather top-k results
        top_labels = sorted(label_scores.items(), key=lambda kv: -kv[1])[:topk]
        rows, cols, data = [], [], []
        for g, score in top_labels:
            rows.append(i)
            cols.append(g)
            data.append(score)

        label_idx_found = next((idx for idx, g in enumerate([g for g, _ in top_labels])
                                if initial_labels[g] in gold_set), -1)
        hit = int(label_idx_found != -1)

        return rows, cols, data, (hit, label_idx_found, matched_cluster_found, gold_set)

    def predict(self, X_text_query, gold_labels=None, topk=10, alpha=0.5, beam_size=10, n_jobs=-1):
        X_query = self.text_encoder.predict(X_text_query).toarray()
        Z = self.Z
        n_layers = len(self.model.hmodel)

        # Precompute augmented label embeddings for each layer (+3 features per layer)
        z_layer = [Z]
        for layer_idx in range(1, n_layers):
            pad_width = 3 * layer_idx
            z_layer.append(np.hstack([Z, np.zeros((Z.shape[0], pad_width), dtype=Z.dtype)]))

        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=10)( # was locky
            delayed(XModel._predict)(
                i, X_query, z_layer, self.model, self.initial_labels,
                gold_labels, topk, alpha, beam_size, n_layers
            )
            for i in range(X_query.shape[0])
        )

        # Aggregate batch results
        rows, cols, data, hits = [], [], [], []
        for r, c, d, h in results:
            rows.extend(r)
            cols.extend(c)
            data.extend(d)
            hits.append(h)

        return csr_matrix((data, (rows, cols)), shape=(X_query.shape[0], Z.shape[0])), hits