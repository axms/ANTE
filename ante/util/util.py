from collections import defaultdict


def get_first_or_none(the_list):
    if len(the_list) == 0:
        return None
    if type(the_list) == set:
        the_list = list(the_list)
    return the_list[0]


def nested_dict():
    return defaultdict(nested_dict)
