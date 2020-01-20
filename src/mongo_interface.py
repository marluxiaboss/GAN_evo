import os
from pymongo import MongoClient


root_password = os.environ['MONGOROOTPASS']
print(root_password)
client = MongoClient(username='root', password=root_password)


def push_to_db(payload, type):
    # payload: {md}

    def check_payload():
        pass

    if type not in ['gan-disc', 'image']:
        raise(Exception("%s is not a supported collection in the database" % type))

    collection = client[type]

    insertion_id = collection.insert_one(payload)

    return insertion_id


def get_from_db(key, type):

    if type not in ['gan-disc', 'image']:
        raise(Exception("%s is not a supported collection in the database" % type))

    collection = client[type]

    return collection.find_one(key)


def update_in_db(key, type, update_payload):

    if type not in ['gan-disc', 'image']:
        raise(Exception("%s is not a supported collection in the database" % type))

    collection = client[type]

    return collection.find_one_and_update(key,
                                          {"$set": update_payload})


def list_data_by_filter(filter, type):

    if type not in ['gan-disc', 'image']:
        raise(Exception("%s is not a supported collection in the database" % type))

    collection = client[type]

    return collection.find(filter)  # that returns a cursor - aka an iterator


if __name__ == "__main__":
    gans_db = client['gan-disc']
    image_db = client['image']
