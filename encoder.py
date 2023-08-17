import numpy as np
"""Use np.log1p instead of normal np.log to assure robust operation with low current"""


def encode_input(input_array):
    return input_array


def encode_output(output_array):
    # log(1E9 * current)
    return np.log1p(np.abs(output_array)) / np.log(10)


def decode_input(input_array):
    return input_array


def decode_output(output_array):
    # exp(output) / 1E9
    return (np.power(10, output_array) - 1) * 1E-9


def decode(input_array, output_array):
    input_return    = input_array
    output_return   = (np.power(10, output_array) - 1) * 1E-9
    return input_return, output_return


def encode(input_array, output_array):
    input_return    = input_array
    output_return   = np.log1p(np.abs(output_array)) / np.log(10)
    return input_return, output_return


def wrap_data(input_array, output_array, training_ratio):
    input_origin, output_origin = encode(input_array, output_array)
    randomized_ndx = np.arange(len(input_origin))
    np.random.shuffle(randomized_ndx)
    input_shuffled = np.array(input_origin)[randomized_ndx]
    output_shuffled = np.array(output_origin)[randomized_ndx]
    training_input = input_shuffled[:round(len(input_shuffled) * training_ratio)]
    training_output = output_shuffled[:round(len(output_shuffled) * training_ratio)]
    validation_input = input_shuffled[round(len(input_shuffled) * training_ratio):]
    validation_output = output_shuffled[round(len(output_shuffled) * training_ratio):]

    training_dataset = [(x, y) for x, y in zip(training_input, training_output)]
    validation_dataset = [(x, y) for x, y in zip(validation_input, validation_output)]

    return training_dataset, validation_dataset
