import json

def json_load(path):
    """
    The script allows users to load a JSON file that maps the class
    values to other category names
    """
    with open(path, 'r') as f:
        cat_to_name = json.load(f)

    print(len(cat_to_name))

    return cat_to_name
