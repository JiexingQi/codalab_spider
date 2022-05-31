#coding=utf8
import os, json, pickle, time
from tqdm import tqdm
import fcntl

from .common_utils import Preprocessor
from .align_tables import align_tables_by_dataset_name

def process_tables(processor, tables_list, used_db_set, output_path=None):
    tables = {}
    for idx, each in tqdm(enumerate(tables_list)):
        if each['db_id'] not in used_db_set:
            continue
        tables[each['db_id']] = processor.preprocess_database(each)
    print('In total, process %d databases .' % (len(tables)))
    # if output_path is not None:
    #     with open (output_path, 'wb') as dump_f:
    #         fcntl.flock(dump_f.fileno(), fcntl.LOCK_EX)
    #         pickle.dump(tables, dump_f)
    return tables

def process_dataset(processor, dataset, tables, dataset_name, mode, output_path_base=None, used_coref=False):
    processed_dataset = []
    # if dataset_name == "cosql":
    #     wfile = open(f"./dataset_files/ori_dataset/cosql_dataset/sql_state_tracking/{mode}_text_list.txt", "w")
    # if used_coref and not os.path.exists(os.path.join(output_path_base, f"{mode}_coref.json")):
    #     wfile = open(os.path.join(output_path_base, f"{mode}_text_list.txt"), "w")
    for idx, entry in tqdm(enumerate(dataset)):
        # if idx > 100:
        #     continue
        if dataset_name in ["spider"]:
            entry = processor.pipeline(entry, tables[entry['db_id']], dataset_name, idx)
        elif dataset_name in ["cosql", "sparc"]:
            entry = processor.pipeline(entry, tables[entry['database_id']], dataset_name, idx)
        else:
            raise NotImplementedError
        # if used_coref and not os.path.exists(os.path.join(output_path_base, f"{mode}_coref.json")):
        #     wfile.write(str(entry['final_preprocessed_text_list'])+"\n")
        processed_dataset.append(entry)
    # with open(os.path.join(output_path_base, f"{mode}.bin"), 'wb') as dump_f:
    #     fcntl.flock(dump_f.fileno(), fcntl.LOCK_EX)
    #     pickle.dump(processed_dataset, dump_f)
    return processed_dataset

# def init_dataset_path(data_base_dir, dataset_name, mode):
    # db_dir = os.path.join(data_base_dir, "ori_dataset", dataset_name, "database")
    # table_data_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "tables.json")
    # table_out_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "tables.bin")
    # if mode == "train":
    #     if dataset_name == "spider":
    #         dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "train_spider.json")
    #     elif dataset_name == "cosql":
    #         db_dir = os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "database")
    #         dataset_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset/sql_state_tracking/", "cosql_train.json")
    #         table_data_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "tables.json")
            
    #     elif dataset_name == "sparc":
    #         dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "train.json")
    #     else:
    #         raise NotImplementedError
    #     # dataset_output_path_base=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "train.bin")
    # elif mode == "dev": 
    #     if dataset_name in ["spider", "sparc"] :
    #         dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "dev.json")
    #     elif dataset_name == "cosql":
    #         db_dir = os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "database")
    #         dataset_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset/sql_state_tracking/", "cosql_dev.json")
    #         table_data_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "tables.json")
            
    #     else:
    #         raise NotImplementedError
    #     # dataset_output_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "dev.bin")
    # else:
    #     raise NotImplementedError
    # dataset_output_path_base=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)
    # if not os.path.exists(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)):
    #     os.makedirs(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name))
    # return db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path_base

def init_dataset_path(data_base_dir, dataset_name, mode):
    # db_dir = os.path.join(data_base_dir, "ori_dataset", dataset_name, "database")
    # table_data_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "tables.json")
    db_dir = os.path.join(data_base_dir, "/database")
    table_data_path=os.path.join(data_base_dir, "/data/sparc/tables.json")
    table_out_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "tables.bin")
    if mode == "train":
        if dataset_name == "spider":
            dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "train_spider.json")
        elif dataset_name == "cosql":
            db_dir = os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "database")
            dataset_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset/sql_state_tracking/", "cosql_train.json")
            table_data_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "tables.json")
        elif dataset_name == "sparc":
            dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "train.json")
        else:
            raise NotImplementedError
        dataset_output_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "train.bin")
    elif mode == "dev": 
        if dataset_name == "spider" :
            is_test_file_exist = os.path.exists("/data/test.json")
            dev_or_test_filepath = "/data/test.json" if is_test_file_exist else "/data/spider/dev.json"
            dataset_path=os.path.join(data_base_dir, dev_or_test_filepath)
        elif dataset_name == "sparc" :
            is_test_file_exist = os.path.exists("/data/sparc/test.json")
            dev_or_test_filepath = "/data/sparc/test.json" if is_test_file_exist else "/data/sparc/dev.json"
            dataset_path=os.path.join(data_base_dir, dev_or_test_filepath)
            table_data_path=os.path.join(data_base_dir, "/data/sparc/tables.json")
            db_dir = os.path.join(data_base_dir, "/data/sparc/database")
        elif dataset_name == "cosql":
            is_test_file_exist = os.path.exists("/cosql_dataset/sql_state_tracking/cosql_test.json")
            dev_or_test_filepath = "/cosql_dataset/sql_state_tracking/cosql_test.json" if is_test_file_exist else "/cosql_dataset/sql_state_tracking/cosql_dev.json"
            db_dir = os.path.join(data_base_dir, "/cosql_dataset/database")
            dataset_path=os.path.join(data_base_dir, dev_or_test_filepath)
            table_data_path=os.path.join(data_base_dir, "/cosql_dataset/tables.json")
        else:
            raise NotImplementedError
        dataset_output_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "dev.bin")
    else:
        raise NotImplementedError
    # if not os.path.exists(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)):
    #     os.makedirs(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name))
    return db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path

def preprocessing_generate_lgerels(data_base_dir, dataset_name, mode, used_coref = False, use_dependency=False):
    db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path_base = init_dataset_path(data_base_dir, dataset_name, mode)
    processor = Preprocessor(dataset_name, db_dir=db_dir, db_content=True)
    
    # loading database and dataset
    print(f"Dataset name: {dataset_name}")
    print(f"Mode: {mode}")
    with open(dataset_path, 'r') as load_f: 
        fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
        dataset = json.load(load_f)
    with open(table_data_path, 'r') as load_f: 
        fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
        tables_list = json.load(load_f)

    print('Firstly, preprocess the original databases ...')
    tables_list = align_tables_by_dataset_name(dataset_name, tables_list)
    print('Tables alignments done...')

    start_time = time.time()
    used_db_set = set()
    for _, entry in tqdm(enumerate(dataset)):
        if dataset_name in ["spider"]:
            used_db_set.add(entry["db_id"])
        elif dataset_name in ["cosql", "sparc"]:
            used_db_set.add(entry["database_id"])
    tables = process_tables(processor = processor, tables_list = tables_list, used_db_set = used_db_set, output_path = table_out_path)
    print('Databases preprocessing costs %.4fs .' % (time.time() - start_time))    
    dataset = process_dataset(processor = processor, dataset = dataset, tables = tables, dataset_name = dataset_name, mode = mode, output_path_base = dataset_output_path_base, used_coref = used_coref)

    dataset_lgesql = dataset
    table_lgesql = tables

    return dataset_lgesql, table_lgesql
    # if not os.path.exists(table_out_path):
    #     with open(table_data_path, 'r') as load_f: 
    #         fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
    #         tables_list = json.load(load_f)
    #     print('Firstly, preprocess the original databases ...')
    #     tables_list = align_tables_by_dataset_name(dataset_name, tables_list)
    #     print('Tables alignments done...')
    #     start_time = time.time()
    #     tables = process_tables(processor, tables_list, table_out_path)
    #     print('Databases preprocessing costs %.4fs .' % (time.time() - start_time))
    # else:
    #     tables = pickle.load(open(table_out_path, 'rb'))
    #     print('Databases has been preprocessed. Use cache.')

    # with open(dataset_path, 'r') as load_f: 
    #     fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
    #     dataset = json.load(load_f)
    # start_time = time.time()

    # if not os.path.exists(os.path.join(dataset_output_path_base, f"{mode}.bin")):
    #     dataset = process_dataset(processor, dataset, tables, dataset_name, mode, dataset_output_path_base, used_coref)
    #     print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))
    # else:
    #     print('Dataset has been preprocessed. Use cache.')
