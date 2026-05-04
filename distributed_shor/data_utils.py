import sys


def merge_data_dict(dict_old, dict_new):
    # https://stackoverflow.com/questions/43797333/how-to-merge-two-nested-dict-in-python
    for key, val in dict_old.items():
        if type(val) == dict:
            if key in dict_new and type(
                dict_new[key] == dict
            ):  # edge case if dict_old[key] is dict but dict_new[key] is value
                # print(f"Merging dictionaries for {key}")
                merge_data_dict(dict_old[key], dict_new[key])
        else:  # if not dictionary prefer new values
            if key in dict_new:
                # print(f"Overriding values for {key}")
                dict_old[key] = dict_new[key]

    for key, val in dict_new.items():
        if not key in dict_old:
            dict_old[key] = val

    sys.stdout.flush()
    return dict_old
