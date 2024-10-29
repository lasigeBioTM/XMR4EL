import os
import pandas as pd

from src.preprocessing.kb import Kb
from src.preprocessing.preprocessing import Preprocessing

kb_mappings = {
        "medic": {
            "filepath": "CTD_diseases.tsv",
            "id_column": "DiseaseID",
            "name_column": "DiseaseName",
            "skiprows": 29
        },
        "ctd_chemicals": {
            "filepath": "CTD_chemicals.tsv",
            "id_column": "ChemicalID",
            "name_column": "ChemicalName",
            "skiprows": 29
        },
        "ctd_genes": {
            "filepath": "CTD_genes.tsv",
            "id_column": "GeneID",
            "name_column": "GeneSymbol",
            "skiprows": 29
        },
        "ncbi_taxon": {
            "filepath": "ncbi_taxon.tsv",
            "id_column": "TaxonID",
            "name_column": "TaxonName",
            "skiprows": 1
        }
    }

kb_filepath = os.path.abspath("data/raw/mesh_data/medic/CTD_diseases.tsv")
id_column = "DiseaseID"
name_column = "DiseaseName"
skip_rows = 29

kb = Kb(filepath=kb_filepath, kb_type='medic', id_column=id_column, name_column=name_column, skip_rows=skip_rows)
kb.load_tsv()
