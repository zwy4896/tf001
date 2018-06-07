from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tf001
import argparse as ap 
import tensorflow as tf 

parser = ap.ArgumentParser()
parser.add_argument('--batch_size', default = 100, type = int, help = 'batch_size')
# 训练步数
parser.add_argument('--train_steps', default = 1000, type = int, help = 'number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])
    # load_data 读取和解析训练集及测试集
    (train_x, train_y), (test_x, test_y) = tf001.load_data()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key = key))

    # 2 hidden layer DNN
    classifier = tf.estimator.DNNClassifier(feature_columns = my_feature_columns,
                                            hidden_units = [10, 10],
                                            n_classes=3)

    # train the model
    classifier.train(input_fn = lambda:tf001.train_input_fn(train_x, train_y, args.batch_size), steps = args.train_steps)
    
    # Evaluate
    eval_result = classifier.evaluate(input_fn = lambda:tf001.evel_input_fn(test_x, test_y, args.batch_size))

    print('\nAccuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Predictons
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {'SepalLength':[6, 5.9, 6.9],
                'SepalWidth':[3.3, 3.0, 3.1],
                'PetalLength':[1.7, 4.2, 5.4],
                'PetalWidth':[0.5, 1.5, 2.1]}

    predictions = classifier.predict(input_fn = lambda:tf001.evel_input_fn(predict_x, labels = None, batch_size = args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pdct, expec in zip(predictions, expected):
        class_id = pdct['class_ids'][0]
        probability = pdct['probabilities'][class_id]

        print(template.format(tf001.SPECIES[class_id], 100 * probability, expec))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)