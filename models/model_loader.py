import pickle

class PKLLoader:

    def load_pkl(self, fname):
        try:
            with open(fname, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {fname}")
        except RuntimeError as e:
            raise RuntimeError(f"Error loading pickle file: {fname}")