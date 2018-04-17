import argparse
import tensorflow as tf
import pandas as pd

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

label_column = 'income_bracket'


def build_model_columns():
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'
        ])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'
        ])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'
        ])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'
        ])

    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)

    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    base_columns = [
        education,
        marital_status,
        relationship,
        workclass,
        occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    return wide_columns, deep_columns


def create_model(model_dir, wide_columns, deep_columns):
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])


def model_input_function(df_data, shuffle):
    return tf.estimator.inputs.pandas_input_fn(
        x=df_data, y=df_data[label_column], shuffle=shuffle)


def train_model(model, df_train, df_test, epochs):
    for n in range(epochs):
        model.train(input_fn=model_input_function(df_train, shuffle=True))

        results = model.evaluate(
            input_fn=model_input_function(df_test, shuffle=False))

        # Display evaluation metrics
        print('Results at epoch', (n + 1))
        print('-' * 30)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


def save_model(model, columns_to_export, export_model):
    feature_spec = tf.feature_column.make_parse_example_spec(columns_to_export)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    model.export_savedmodel(export_model, serving_input_fn)


def build_and_train_model(model_dir, train_data, test_data, export_model):
    wide_columns, deep_columns = build_model_columns()
    model = create_model(model_dir, wide_columns, deep_columns)
    df_train = pd.read_csv(
        train_data, names=_CSV_COLUMNS, skipinitialspace=True)
    df_test = pd.read_csv(test_data, names=_CSV_COLUMNS, skipinitialspace=True)
    train_model(model, df_train, df_test, 10)
    save_model(model, wide_columns + deep_columns, export_model)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        help='The directory where the model checkpoints should be stored')
    parser.add_argument(
        '--train_data',
        dest='train_data',
        type=str,
        help='The csv file containing the train data')
    parser.add_argument(
        '--test_data',
        dest='test_data',
        type=str,
        help='The csv file containing the test data')
    parser.add_argument(
        '--export_model',
        dest='export_model',
        type=str,
        help='The directory where the model should be stored')
    args = parser.parse_args()
    build_and_train_model(args.model_dir, args.train_data, args.test_data,
                          args.export_model)
