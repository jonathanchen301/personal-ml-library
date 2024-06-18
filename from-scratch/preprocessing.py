
class Preprocessing:
    def __init__(self):
        pass

    def standardize(self, X):
        """
        Returns the standardized version of an input numpy array X by subtracting
        from each element the mean of the array and dividing by the standard deviation
        of the array

        X: numpy array
        """

        return (X - X.mean()) / X.std()

