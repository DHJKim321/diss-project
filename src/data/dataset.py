

class DataSet:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def set_data(self, new_data):
        self.data = new_data

    def __repr__(self):
        return f"DataSet(data={self.data})"

    def __len__(self):
        return len(self.data)