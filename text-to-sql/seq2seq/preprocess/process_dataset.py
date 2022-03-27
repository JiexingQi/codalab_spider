#coding=utf8
import os, json, pickle, time
from tqdm import tqdm
import fcntl

from .common_utils import Preprocessor
from .align_tables import align_tables_by_dataset_name


def process_tables(processor, tables_list, output_path=None):
    tables = {}
    for idx, each in tqdm(enumerate(tables_list)):
        tables[each['db_id']] = processor.preprocess_database(each)
    print('In total, process %d databases .' % (len(tables)))
    # if output_path is not None:
    #     with open (output_path, 'wb') as dump_f:
    #         fcntl.flock(dump_f.fileno(), fcntl.LOCK_EX)
    #         pickle.dump(tables, dump_f)
    return tables

def process_dataset(processor, dataset, tables, dataset_name, output_path=None):
    processed_dataset = []
    for idx, entry in tqdm(enumerate(dataset)):
        # if idx > 100:
        #     continue
        if dataset_name in ["spider"]:
            entry = processor.pipeline(entry, tables[entry['db_id']], dataset_name, idx)
        elif dataset_name in ["cosql", "sparc"]:
            entry = processor.pipeline(entry, tables[entry['database_id']], dataset_name, idx)
        else:
            raise NotImplementedError
        processed_dataset.append(entry)
    # with open(output_path, 'wb') as dump_f:
    #     fcntl.flock(dump_f.fileno(), fcntl.LOCK_EX)
    #     pickle.dump(processed_dataset, dump_f)
    return processed_dataset

def init_dataset_path(data_base_dir, dataset_name, mode):
    # db_dir = os.path.join(data_base_dir, "ori_dataset", dataset_name, "database")
    # table_data_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "tables.json")
    db_dir = os.path.join(data_base_dir, "/database")
    table_data_path=os.path.join(data_base_dir, "/data/tables.json")
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
        if dataset_name in ["spider", "sparc"] :
            # dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "dev.json")
            dataset_path=os.path.join(data_base_dir, "/data/dev_mini.json")
        elif dataset_name == "cosql":
            db_dir = os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "database")
            dataset_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset/sql_state_tracking/", "cosql_dev.json")
            table_data_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "tables.json")
        else:
            raise NotImplementedError
        dataset_output_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "dev.bin")
    else:
        raise NotImplementedError
    # if not os.path.exists(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)):
    #     os.makedirs(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name))
    return db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path

def preprocessing_generate_lgerels(data_base_dir, dataset_name, mode):
    db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path = init_dataset_path(data_base_dir, dataset_name, mode)
    processor = Preprocessor(dataset_name, db_dir=db_dir, db_content=True)
    
    # loading database and dataset
    print(f"Dataset name: {dataset_name}")
    print(f"Mode: {mode}")
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
    with open(table_data_path, 'r') as load_f: 
        fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
        tables_list = json.load(load_f)
    print('Firstly, preprocess the original databases ...')
    tables_list = align_tables_by_dataset_name(dataset_name, tables_list)
    print('Tables alignments done...')
    start_time = time.time()
    tables = process_tables(processor, tables_list, table_out_path)
    print('Databases preprocessing costs %.4fs .' % (time.time() - start_time))    

    with open(dataset_path, 'r') as load_f: 
        fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
        dataset = json.load(load_f)
    # start_time = time.time()
    # if not os.path.exists(dataset_output_path):
    #     dataset = process_dataset(processor, dataset, tables, dataset_name, dataset_output_path)
    #     print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))
    # else:
    #     print('Dataset has been preprocessed. Use cache.')
    dataset = process_dataset(processor, dataset, tables, dataset_name, dataset_output_path)

    dataset_lgesql = dataset
    table_lgesql = tables

    return dataset_lgesql, table_lgesql
