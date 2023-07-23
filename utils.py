def load_dict(file_path  , required_index=0):
    dict_item = dict()
    with open(file_path) as myfile:
        for index , line in enumerate(myfile):
            value = line.strip().split("\t")[required_index]
            dict_item[value] = int(index)
    return dict_item