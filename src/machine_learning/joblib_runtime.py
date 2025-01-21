from joblib import parallel_backend

def force_multi_core_processing_clustering_models(model, X_train):
    with parallel_backend('threading', n_jobs=-1):
         model.fit(X_train)
    return model 

def force_multi_core_processing_linear_models(model, X_train, Y_train):
    with parallel_backend('threading', n_jobs=-1):
        model.fit(X_train, Y_train)
    return model