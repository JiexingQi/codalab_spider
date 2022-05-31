import spacy
from spacy.tokens import Doc
import coreferee
import json
from tqdm import tqdm
import os
import fcntl
import stanza

nlp_tokenize = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized = False, use_gpu=True)
nlp_pretokenize = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized = True,use_gpu=True)

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)

def init_nlp():
    nlp = spacy.load('en_core_web_trf')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    nlp.add_pipe('coreferee')
    return nlp

def find_turn_idx(id, turn_list, len_text_list):
    total_length = 0
    for idx, length in enumerate(len_text_list):
        total_length += length
        if id < total_length and id >= total_length-length:
            return turn_list[idx], id-total_length+length
    return turn_list[0], id

def text_list2coref_json(dataset, output_path, mode, nlp):
    new_res=[]
    for idx, entry in tqdm(enumerate(dataset)):
        final_preprocessed_text_list = dataset[idx]
        text_list = " ".join([i for item in final_preprocessed_text_list for i in item[1]])
        turn_list = [item[0] for item in final_preprocessed_text_list]
        len_text_list = [item[2] for item in final_preprocessed_text_list]
        
        doc = nlp(text_list)
        coref_dict = {}
        for chain in doc._.coref_chains:
            key = chain.index
            used_turn = set()
            coref_dict[key] = {}
            coref_dict[key]["group"] = []
            for li in [list(_) for _ in chain]:
                new_list = []
                for idx in li:
                    item = find_turn_idx(idx, turn_list, len_text_list)
                    item_dict = {"turn": item[0], "position": item[1], "ori": idx}
                    used_turn.add(item[0])
                new_list.append(item_dict)
                coref_dict[key]["group"].append(new_list)
            coref_dict[key]["used_turn"] = list(used_turn)
        new_entry = {}
        new_entry["coref"] = coref_dict
        new_entry["text_list"] = text_list
        new_res.append(new_entry)
    with open(os.path.join(output_path, f'{mode}_coref_test.json'),"w") as dump_f:
        json.dump(new_res,dump_f) 



def pipeline(entry: dict, db: dict, dataset_name: str, data_idx: int):
        entry["final_preprocessed_text_list"] = []
        if dataset_name in ["cosql", "sparc"]:
            entry =  multi_turn_pipeline(entry, db, dataset_name, data_idx)
        elif dataset_name in ["spider"]:
            entry = single_turn_pipeline(entry, db, dataset_name, data_idx)
        return entry


def quote_normalization(dataset_name, data_idx, question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks, change_marks = [], ["'", '"', '`', '‘', '’', '“', '”', "‘‘", "’’"], ['``', "''"]
    idx = 0
    while idx < len(question):
        tok=question[idx]
        # if dataset_name == "cosql" and data_idx == 441 and tok=="''voluptatem":
        #     new_question.append(tok) 
        #     idx += 1
        #     new_question.append(question[idx])
        #     idx += 1
        #     continue
        if dataset_name == "sparc":
            tok = " ".join(tok.split("\ufeff")).strip()
        for mark in change_marks:
            tok = tok.replace(mark, "\"")
        if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
            new_question += [tok[0], tok[1:-1], tok[-1]]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += [tok[0], tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], tok[-1]]
        elif tok in quotation_marks:
            new_question.append(tok)
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
                new_question += [tok[0], tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
        idx += 1
    return new_question

def preprocess_question(entry: dict, db: dict, turn: str, dataset_name: str, data_idx: int):
    """ Tokenize, lemmatize, lowercase question"""
    if turn == "-1":
        question = " ".join(quote_normalization(dataset_name, data_idx, entry["question_toks"]))
        entry["processed_text_list"] = [question]
    elif "#" in turn:
        parent_idx, son_idx = turn.split("#")
        question = entry["processed_text_list"][int(parent_idx)][int(son_idx)]
    else:
        question = entry["processed_text_list"][int(turn)][0]
    question = question.strip()
    if turn == "0":
        doc = nlp_tokenize(question)
    else:
        doc = nlp_pretokenize(question)
    raw_toks = [w.text.lower() for s in doc.sentences for w in s.words]
    toks = [w.lemma.lower() for s in doc.sentences for w in s.words]
    entry[f'raw_question_toks_{turn}'] = raw_toks
    entry[f'ori_toks_{turn}'] = [w.text for s in doc.sentences for w in s.words]
    entry[f'processed_question_toks_{turn}'] = toks
    entry["final_preprocessed_text_list"].append([turn, [w.text for s in doc.sentences for w in s.words], len([w.text for s in doc.sentences for w in s.words])])
    return entry

def init_dataset_path(data_base_dir, dataset_name, mode):
    db_dir = os.path.join(data_base_dir, "ori_dataset", dataset_name, "database")
    table_data_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "tables.json")
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
        # dataset_output_path_base=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "train.bin")
    elif mode == "dev": 
        if dataset_name in ["spider", "sparc"] :
            dataset_path=os.path.join(data_base_dir, "ori_dataset", dataset_name, "dev.json")
        elif dataset_name == "cosql":
            db_dir = os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "database")
            dataset_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset/sql_state_tracking/", "cosql_dev.json")
            table_data_path=os.path.join(data_base_dir, "ori_dataset", "cosql_dataset", "tables.json")
            
        else:
            raise NotImplementedError
        # dataset_output_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "dev.bin")
    else:
        raise NotImplementedError
    dataset_output_path_base=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)
    if not os.path.exists(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)):
        os.makedirs(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name))
    return db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path_base


def process_dataset(dataset, dataset_name, mode):
    text_list = []
    for idx, entry in tqdm(enumerate(dataset)):
        if idx > 100:
            continue
        if dataset_name in ["spider"]:
            entry = pipeline(entry, {}, dataset_name, idx)
        elif dataset_name in ["cosql", "sparc"]:
            entry = pipeline(entry, {}, dataset_name, idx)
        else:
            raise NotImplementedError
        text_list.append(entry['final_preprocessed_text_list'])
    return text_list

def multi_turn_pipeline(entry: dict, db: dict, dataset_name: str, data_idx: int):
    """ db should be preprocessed """
    entry["text_list"] = []
    question = entry['final']['utterance']
    entry["text_list"].append(question)
    entry["text_list"].extend([" ".join(quote_normalization(dataset_name, data_idx, entry["interaction"][turn]["utterance_toks"])) for turn in range(len(entry['interaction']))])
    entry["processed_text_list"] = []
    for q in entry["text_list"]:
        q = q.split("|")
        entry["processed_text_list"].append(q)
    for idx, turn_item in enumerate(entry['processed_text_list']):
        idx = str(idx)
        if len(turn_item) > 1:
            for i in range(len(turn_item)):
                multi_idx = idx + "#"+str(i)
                if i== 0 :
                    entry[multi_idx] = turn_item[1].count("|")+1
                entry = preprocess_question(entry, db, multi_idx, dataset_name, data_idx)
        else:
            entry = preprocess_question(entry, db,  idx, dataset_name, data_idx)
    return entry

def single_turn_pipeline(entry: dict, db: dict, dataset_name: str, data_idx: int):
    entry = preprocess_question(entry, db, "-1", dataset_name, data_idx)
    return entry

def getText(data_base_dir, dataset_name, mode):
    _, _, _, dataset_path, _ = init_dataset_path(data_base_dir, dataset_name, mode)
    
    with open(dataset_path, 'r') as load_f: 
        fcntl.flock(load_f.fileno(), fcntl.LOCK_EX)
        dataset = json.load(load_f)
    text_list = process_dataset(dataset, dataset_name, mode)
    return text_list

def test2assertSame(data_base_dir, dataset_name, mode):
    _, _, _, dataset_path, dataset_output_path_base = init_dataset_path(data_base_dir, dataset_name, mode)
    with open(os.path.join(dataset_output_path_base, f"{mode}_text_list.txt"), 'r') as load_f:
       true_dataset = load_f.readlines()
    with open(os.path.join(dataset_output_path_base, f"{mode}_text_list_test.txt"), 'r') as load_f:
       cur_dataset = load_f.readlines()
    for idx, line in enumerate(cur_dataset):
        assert line == true_dataset[idx], f"{line}, {true_dataset[idx]}"

def main():
    mode_list = ["train", "dev"]
    dataset_name_list = ["spider", "cosql"]
    data_base_dir = "dataset_files/"
    nlp = init_nlp()
    for dataset_name in dataset_name_list:
        for mode in mode_list:
            output_path = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)
            text_list = getText(data_base_dir, dataset_name, mode)
            text_list2coref_json(text_list, output_path, mode, nlp)

main()