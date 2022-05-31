from .process_dataset import preprocessing_generate_lgerels
from .lgerels2t5rels import preprocessing_lgerels2t5rels


def preprocess_by_dataset(data_base_dir, dataset_name, t5_processed, mode):
    dataset_lgesql, table_lgesql = preprocessing_generate_lgerels(data_base_dir, dataset_name, mode)
    _, relations=preprocessing_lgerels2t5rels(data_base_dir, dataset_name, t5_processed, dataset_lgesql, table_lgesql, mode)
    return relations