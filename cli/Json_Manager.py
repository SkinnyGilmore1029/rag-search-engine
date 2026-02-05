import json

def load_Json_to_dict(file_path: str) -> dict:
    """Load a JSON file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
 
"""
for every dictionary in the list:
    


"""