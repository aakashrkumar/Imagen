import ray

class DataFetcher:
    def __init__(self, data):
        self.data = data

    def fetch(self):
        return self.data