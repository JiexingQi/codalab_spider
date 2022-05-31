from .process_dataset import preprocessing_generate_lgerels
from .lgerels2t5rels import preprocessing_lgerels2t5rels
from .lgerels2t5rels_changeOrder import preprocessing_lgerels2t5rels_changeOrder
from .get_relation2id_dict import get_relation2id_dict


def preprocess_by_dataset(data_base_dir, dataset_name, t5_processed, mode, edge_type="Default", use_coref=False, use_dependency = False):
    # edgeType: Default, MinType
    # edgeType = "Default"
    # use_coref = True
    # use_dependency =True
    dataset_lgesql, table_lgesql = preprocessing_generate_lgerels(data_base_dir = data_base_dir, dataset_name = dataset_name, mode = mode, used_coref = use_coref, use_dependency = use_dependency)
    # _, relations=preprocessing_lgerels2t5rels(data_base_dir, dataset_name, t5_processed, mode, edgeType, use_coref)
    _, relations = preprocessing_lgerels2t5rels_changeOrder(data_base_dir = data_base_dir, dataset_name = dataset_name, t5_processed = t5_processed, dataset_lgesql = dataset_lgesql, table_lgesql = table_lgesql, mode = mode, edgeType = edge_type, use_coref = use_coref, use_dependency = use_dependency)
    return relations