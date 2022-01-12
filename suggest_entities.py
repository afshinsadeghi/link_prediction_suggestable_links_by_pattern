import numpy as np
import os
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Generating possible links from datasets',
        usage='python  suggest_entities.py --path <path to dataset folder>'
    )
    parser.add_argument('--path', default="./Dataset/",
                        type=str, help='path to dataset folder')
    parser.add_argument('--type', default="common_predicate",
                        type=str, help='type of suggestable links')
    return parser.parse_args(args)


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def write_triple(file_path, triples, entity2id_inv, relation2id_inv):
    '''
    write triples and map them into ids.
    '''
    with open(file_path, "w") as fin:
        for line in triples:
            h, r, t = line
            # triples.append(entity2id_inv[h], relation2id_inv[r],
            # entity2id_inv[t])
            fin.write(
                entity2id_inv[h] + '\t' + relation2id_inv[r] + '\t' +
                entity2id_inv[t] + '\n')


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def find_suggestable_triples_on_pattern_of_r_t_in_common(train_data, input_directory):
    # print("------------------------+-----------------------")
    #print("extracting suggestable triples for the dataset relations:", input_directory)

    train_dic = {}
    train_dic_r = {}
    train_dic_h = {}
    train_dic = {(a, c): b for a, b, c in train_data}
    train_dic_h_r = {}
    train_dic_t_r = {}
    train_dic_inv = {(c, a): b for a, b, c in train_data}
    h_r_t_dic = {(a, b, c): 1 for a, b, c in train_data}

    for a, b, c in train_data:
        temp = train_dic_r.get(b, None)
        if temp is None:
            train_dic_r[b] = [(a, c)]
        else:
            temp.append((a, c))
            train_dic_r[b] = temp

        temp1 = train_dic_h.get(a, None)
        if temp1 is None:
            train_dic_h[a] = [(b, c)]
        else:
            temp1.append((b, c))
            train_dic_h[a] = temp1

        temp2 = train_dic_h_r.get((a, b), None)
        if temp2 is None:
            train_dic_h_r[(a, b)] = [c]
        else:
            temp2.append(c)
            train_dic_h_r[(a, b)] = temp2

        temp3 = train_dic_t_r.get((c, b), None)
        if temp3 is None:
            train_dic_t_r[(c, b)] = [a]
        else:
            temp3.append(a)
            train_dic_t_r[(c, b)] = temp3
    # 1. first filter out those h that has more than threshod relations. (for example 2) (their h lists are longer than threshold)
    # (in train_dic_t_r those lists which are longer than threshhold (for example 2), take them!)
    #
    # 2.from them, those h that have same r_t (one common relation and tail) extend their other r_t to each other
    # to get those h that have use the value of train_dic_t_r. get those that have more than one h and take the values of the dict

    # step 2 can be extended to those that have more than one common relation.
    new_triples = {}
    # step 1
    threshold = 1
    dic_h_filtered = {
        key: value for key, value in train_dic_h.items() if len(value) > threshold}

    those_h_from_same_r_t = {
        key: value for key, value in train_dic_t_r.items() if len(value) > threshold}
    for key, value in those_h_from_same_r_t.items():
        for v1 in value:
            # print(key)
            if dic_h_filtered.get(v1, None) is not None:
                for v2 in value:
                    if v1 != v2 and dic_h_filtered.get(v2, None) is not None:
                        # step 2
                        # generate new triples here
                        # get r_t of h1 add them to h2 as new triples.
                        temp1 = train_dic_h.get(v1, None)
                        temp2 = train_dic_h.get(v2, None)
                        for r_t in temp1:
                            if h_r_t_dic.get((v2, r_t[0], r_t[1]), None) is None:
                                new_triples[(v2, r_t[0], r_t[1])] = 1
                        for r_t in temp2:
                            if h_r_t_dic.get((v1, r_t[0], r_t[1]), None) is None:
                                new_triples[(v1, r_t[0], r_t[1])] = 1

    print("-------------")

    return list(new_triples.keys())


def extract_pattern(args):
    input_directory = args.path
    print("-------------------------------------------------------")
    print("testing dir: " + input_directory)
    # get all files' and folders' names in the current directory, ignore hidder folders
    db_dir = listdir_nohidden(input_directory)
    filenames = [_ for _ in db_dir]
    print("extracting suggestable triples for link prediction:", input_directory)

    #print("(please give folder of folders like ./data/ where ./data/data.txt is inside it)")
    result = []
    # for filename in filenames:  # loop through all the files and folders
    # check whether the current object is a folder
    if os.path.isdir(os.path.join(input_directory)):
        result.append(input_directory)

    for dataset_name in result:
        if not os.path.exists(dataset_name + '/entities.dict'):
            print("entities.dict does not exist. first run makedict_for_pattern_rel_dbs.py to generate that. folder: " +
                  + dataset_name)
            exit()

        with open(dataset_name + '/entities.dict') as fin:
            entity2id = dict()
            entity2id_inv = {}
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)
                entity2id_inv[int(eid)] = entity

        with open(dataset_name + '/relations.dict') as fin:
            relation2id = {}
            relation2id_inv = {}
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)
                relation2id_inv[int(rid)] = relation

        train_data_ = read_triple(
            dataset_name + "/train.txt", entity2id, relation2id)
        # train_data_ = np.array(train_data_)[:, [0, 2, 1]]  # it's column must be in shape [entity, entity,relation]

        # test_data_ = read_triple(
        #    dataset_name + "/test.txt", entity2id, relation2id)

        out_dir_path = dataset_name  # + "/suggested_triples.txt"

        # making it as h t r
        train_dic = {(a, b, c): 1 for a, b, c in train_data_}
        # making it as h t r
        #train_dic_inv = {(c, b, a): 1 for a, b, c in train_data_}
        #test_dic = {(a, b, c): 1 for a, b, c in test_data_}
        #test_dic_inv = {(c, b, a): 1 for a, b, c in test_data_}

        train_dic_r = {}
        train_dic_r_inv = {}
        #test_dic_r = {}
        #test_dic_r_inv = {}
        for a, r, c in train_data_:
            if train_dic_r.get(r, None) is None:
                train_dic_r[r] = [(a, r, c)]
            else:
                r_list = train_dic_r[r]
                r_list.append((a, r, c))
                train_dic_r[r] = r_list

        for a, r, c in train_data_:
            if train_dic_r_inv.get(r, None) is None:
                train_dic_r_inv[r] = [(c, r, a)]
            else:
                r_list = train_dic_r_inv[r]
                r_list.append((c, r, a))
                train_dic_r_inv[r] = r_list

        # for a, r, c in test_data_:
        #    if test_dic_r.get(r, None) is None:
        #        test_dic_r[r] = [(a, r, c)]
        #    else:
        #        r_list = test_dic_r[r]
        #        r_list.append((a, r, c))
        #        test_dic_r[r] = r_list

        # for a, r, c in test_data_:
        #    if test_dic_r_inv.get(r, None) is None:
        #        test_dic_r_inv[r] = [(c, r, a)]
        #    else:
        #        r_list = test_dic_r_inv[r]
        #        r_list.append((c, r, a))
        #        test_dic_r_inv[r] = r_list

        out_dir_path = out_dir_path + "/"

        suggested_triples = find_suggestable_triples_on_pattern_of_r_t_in_common(
            train_data_, out_dir_path)
        head_triples_path = out_dir_path + "suggested_triples.txt"
        write_triple(head_triples_path, suggested_triples,
                     entity2id_inv, relation2id_inv)

        return


print("example run: python suggest_entities.py --path ./wn18rr/")
# python suggest_entities.py --path ./Dataset/fb15k/
args_ = parse_args()
extract_pattern(args_)

# requires python 3.9
# source ~/miniconda3/bin/activate
