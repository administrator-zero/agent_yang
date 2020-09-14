class ConvUtil(object):
    @staticmethod
    def get_patch(input_array, i, j, filter_width, filter_height, stride):
        """
        Get region from input tensor and support 2 dmi and 3 dim.
        """
        start_i = i * stride
        start_j = j * stride
        if input_array.ndim == 2:
            return input_array[start_i: start_i + filter_height, start_j: start_j + filter_width]
        elif input_array.ndim == 3:
            return input_array[:, start_i: start_i + filter_height, start_j: start_j + filter_width]
