from loader import Loader


class LabelLoader(Loader):
    """
    The label loader of MNIST
    """
    def load(self):
        """
        Load all data and get all labels.
        """
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    @staticmethod
    def norm(label):
        """
        Turn label to 10 dim vector.
        """
        label_vec = []
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vec.append(1)
            else:
                label_vec.append(0)
        return label_vec
