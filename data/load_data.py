import pickle


def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data.get('data'), data.get('labels')
