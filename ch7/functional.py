# Leonard R. Kosta Jr.

import numpy as np
from keras import Input, layers
from keras.models import Model
from keras import applications


def test_functional():
    """An example functional model."""
    input_tensor = Input(shape=(32,))
    dense = layers.Dense(32, activation='relu')
    output_tensor = dense(input_tensor)


def multi_input():
    """Runs a question-answer model that takes two inputs and produces
    one output. The inputs are a question and a text, and the output is
    a softmax over the vocabulary."""
    # TODO this whole example from the book seems untested.
    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500
    text_input = Input(shape=(None,), dtype='int32', name='text')
    # TODO the embedding arguments seem backward.
    embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
    encoded_text = layers.LSTM(32)(embedded_text)
    question_input = Input(shape=(None,),
                           dtype='int32',
                           name='questions')
    # TODO the embedding arguments seem backward.
    embedded_question = layers.Embedding(32, question_vocabulary_size)(
        question_input)
    encoded_question = layers.LSTM(16)(embedded_question)
    concatenated = layers.concatenate([encoded_text, encoded_question],
                                      axis=-1)
    answer = layers.Dense(answer_vocabulary_size, activation='softmax')(
        concatenated)
    model = Model([text_input, question_input], answer)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    num_samples = 1000
    max_length = 100
    text = np.random.randint(1, text_vocabulary_size, size=(num_samples,
                                                            max_length))
    question = np.random.randint(1, question_vocabulary_size, size=(num_samples,
                                                                    max_length))
    answers = np.random.randint(0, 1, size=(num_samples,
                                            answer_vocabulary_size))
    # The following are identical.
    #model.fit([text, question], answers, epochs=10, batch_size=128)
    model.fit({'text': text, 'question': question}, answers,
              epochs=10, batch_size=128)


def multi_output():
    """Runs a multi-output model that predicts a person's age, gender,
    and income level from social media posts."""
    vocabulary_size = 50000
    num_income_groups = 10
    posts_input = Input(shape=(None,), dtype='int32', name='posts')
    embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
    x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    age_prediction = layers.Dense(1, name='age')(x)
    income_prediction = layers.Dense(num_income_groups,
                                     activation='softmax',
                                     name='income')(x)
    gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
    model = Model(posts_input,
                  [age_prediction, income_prediction, gender_prediction])
    # The following are identical.
    #model.compile(optimizer='rmsprop',
    #    loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
    #model.compile(optimizer='rmsprop',
    #              loss={'age': 'mse',
    #                    'income': 'categorical_crossentropy',
    #                    'gender': 'binary_crossentropy'})
    # Now, with loss weighting.
    model.compile(optimizer='rmsprop',
                  loss={'age': 'mse',
                        'income': 'categorical_crossentropy',
                        'gender': 'binary_crossentropy'},
                  loss_weights={'age': 0.25,
                                'income': 1.,
                                'gender': 10.})
    # This isn't real data.
    posts = age_targets = income_targets = gender_targets = None
    model.fit(posts, {'age': age_targets,
                      'income': income_targets,
                      'gender': gender_targets},
              epochs=10, batch_size=64)


def inception_module():
    """Runs an inception module on the input."""
    # This isn't real data.
    x = None
    branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
    branch_b = layers.Conv2D(128, 1, activation='relu')(x)
    branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
    branch_c = layers.AveragePooling2D(3, strides=2)(x)
    branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
    branch_d = layers.Conv2D(128, 1, activation='relu')(x)
    branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
    branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
    output = layers.concatenate([branch_a, branch_b, branch_c, branch_d],
                                axis=-1)


def residual_layers():
    """Runs a model with residual layers."""
    # This isn't real data.
    x = None
    y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
    y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
    y = layers.add([y, x])


def weight_sharing():
    """Runs a model that shares weights on two inputs."""
    # This isn't real data.
    left_data = right_data = targets = None
    lstm = layers.LSTM(32)
    left_input = Input(shape=(None, 128))
    left_output = lstm(left_input)
    right_input = Input(shape=(None, 128))
    right_output = lstm(right_input)
    merged = layers.concatenate([left_output, right_output], axis=-1)
    predictions = layers.Dense(1, activation='sigmoid')(merged)
    model = Model([left_input, right_input], predictions)
    model.fit([left_data, right_data], targets)


def model_as_layer():
    """Runs a model that has a model as a layer."""
    xception_base = applications.Xception(weights=None, include_top=False)
    left_input = Input(shape=(250, 250, 3))
    right_input = Input(shape=(250, 250, 3))
    left_features = xception_base(left_input)
    right_features = xception_base(right_input)
    merged_features = layers.concatenate([left_features, right_features],
                                         axis=-1)


def main():
    """Runs the program."""
    multi_input()


if __name__ == '__main__':
    main()
