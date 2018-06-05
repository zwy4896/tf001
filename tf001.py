import pandas as pd 
import tensorflow as tf

# 训练集和测试集合url
TRAIN_URL = "htp://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# 定义x值，sepal length, petal length…………
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

# y值=labels
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Download data
def download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    return train_path, test_path

# Load & parse the csv files using pandas
def load_data(label_name = 'Species'):
    train_path, test_path = download()

    # 这里，header = 0， csv第0行？
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)

    # pop again!!!
    train_x, train_y = train, train.pop(label_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(label_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    data_set = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    data_set = data_set.shuffle(1000).repeat().batch(batch_size)

    return data_set

def evel_input_fn(features, labels, batch_size):
    # For evaluation || prediction
    features = dict(features)

    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    data_set = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size must not be None!"
    data_set = data_set.batch(batch_size)

    return data_set

CSV_TYPES = [[0, 0], [0,0], [0,0], [0]]

def parse_line(line):
    fields = tf.decode_csv(line, record_defaults = CSV_TYPES)

    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop('Species')

    return features, label

def csv_input_fn(csv_path, batch_size):
    data_set = tf.data.TextLineDataset(csv_path).skip(1)
    data_set = data_set.map(parse_line)

    data_set = data_set.shuffle(1000).repeat().batch(batch_size)

    return data_set