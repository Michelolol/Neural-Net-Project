from typing import Tuple
from neural import *


def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    #print(tokens)
    out = float(tokens[8].strip())
    output = [out]
    #print(output)

    inpt = [float(x) for x in tokens[1:8]]
    # print(inpt)
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data


with open("FixedGunData.csv", "r") as f:
    f.readline()
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# for line in training_data:
#     print(line)


td = normalize(training_data)

for line in td:
    print(line)

nn = NeuralNet(7, 3, 1)
nn.train(td, iters=10000, print_interval=1000, learning_rate=0.1)

for i in nn.test_with_expected(td):
    print(f"desired: {i[1]}, actual: {i[2]}")

test_data = [
    [10000, 10000, 10000, 10000, 40000, 10000, 50000], #biased unre
    [0, 0, 0, 0, 0, 0, 0], #biased restrict
    [33000.0,0.0,0.0,19.29,2.37,0.68,0.0], #idk which

]


print(f"case 1: {test_data[0]} evaluates to: {nn.evaluate(test_data[0])}")
print(f"case 2: {test_data[1]} evaluates to: {nn.evaluate(test_data[1])}")
print(f"case 3: {test_data[2]} evaluates to: {nn.evaluate(test_data[2])}")