import os
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import time
import sys
import numpy as np

# Redirect output to a file
with open('output_train_4.txt', 'w') as f:
    sys.stdout = f

    print("hello world!!")
    # Local directory with logs
    # train_directory = "D:/Siddhant/Masters Project/Dataset/Process Discovery Contest 2023_1_all/Training Logs"

    # Parameters
    max_sequence_length = 32
    batch_size = 32
    epochs = 50

    # Extracting concept names
    def extract_concept_names(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        concept_names = []
        for trace in root.iter('trace'):
            for event in trace.iter('event'):
                for string in event.iter('string'):
                    if string.attrib.get('key') == 'concept:name':
                        concept_names.append(string.attrib['value'])
            # end of case token
            concept_names.append('<END>')

        return concept_names

    # Function to read and combine logs from a list of files
    def read_and_combine_logs(log_files):
        combined_activities = []
        for log_path in log_files:
            concept_names = extract_concept_names(log_path)
            combined_activities.extend(concept_names)
        return combined_activities

    # Convert activities to categorical codes
    def encode_activities(activities, activity_to_index=None):
        combined_activities = pd.Series(activities).astype('category')
        if activity_to_index is None:
            activity_to_index = {activity: index for index, activity in enumerate(combined_activities.cat.categories)}
        return [activity_to_index[activity] for activity in combined_activities], activity_to_index

    # Example training sequence generator
    def sequence_generator(activity_codes, batch_size, max_sequence_length, num_classes):
        while True:
            sequences = []
            next_activities = []

            for i in range(0, len(activity_codes) - max_sequence_length):
                if activity_codes[i + max_sequence_length] == activity_codes[activity_codes == '<END>']:
                    continue  # Skip end-of-case tokens for next activity prediction

                seq = activity_codes[i:i + max_sequence_length]
                next_act = activity_codes[i + max_sequence_length]

                sequences.append(seq)
                next_activities.append(next_act)

                if len(sequences) == batch_size:
                    X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
                    y = tf.keras.utils.to_categorical(next_activities, num_classes=num_classes)
                    yield X, y
                    sequences = []
                    next_activities = []

    # CNN model
    def build_cnn_model(input_length, num_classes):
        model = Sequential()
        model.add(Embedding(input_dim=num_classes, output_dim=128, input_length=input_length))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        return model

    # Function to train and save the model
    def train_and_save_cnn_model(log_name, encoded_training_activities, activity_to_index, max_sequence_length=32, epochs=50, batch_size=32):
        num_classes = len(activity_to_index)
        print(f"No of Unique classes = {num_classes}")
        model = build_cnn_model(max_sequence_length, num_classes)

        steps_per_epoch = len(encoded_training_activities) // batch_size

        dataset = tf.data.Dataset.from_generator(
            lambda: sequence_generator(encoded_training_activities, batch_size, max_sequence_length, num_classes),
            output_signature=(
                tf.TensorSpec(shape=(batch_size, max_sequence_length), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.int32)
            )
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)

        # Define the training function with early stopping
        def train_stateful_model(model, dataset, steps_per_epoch, epochs, callbacks):
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                epoch_start_time = time.time()
                print(f'Starting Epoch {epoch + 1}/{epochs}')
                #model.reset_states()  # Reset states at the beginning of each epoch

                total_loss = 0
                total_accuracy = 0

                for step, (X, y) in enumerate(dataset.take(steps_per_epoch)):
                    step_start_time = time.time()
                    loss, accuracy = model.train_on_batch(X, y)
                    total_loss += loss
                    total_accuracy += accuracy
                    print(f"Step {step + 1}/{steps_per_epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Time: {time.time() - step_start_time:.2f} seconds")

                avg_loss = total_loss / steps_per_epoch
                avg_accuracy = total_accuracy / steps_per_epoch
                print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f} seconds - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

                # callbacks
                logs = {'loss': avg_loss, 'accuracy': avg_accuracy}
                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs)

                # Check early stopping conditions
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping.patience:
                    print(f"Early stopping triggered after {patience_counter} epochs with no improvement.")
                    if early_stopping.restore_best_weights:
                        model.set_weights(early_stopping.best_weights)
                    break

                # Save the best weights
                early_stopping.best_weights = model.get_weights()

                # Clear the TensorFlow session to free up memory
                tf.keras.backend.clear_session()

        callbacks = [early_stopping]
        for callback in callbacks:
            callback.set_model(model)
            callback.on_train_begin()

        # Train the model
        train_stateful_model(model, dataset, steps_per_epoch, epochs, callbacks)

        # Extract and save probability distributions
        probability_distributions = []
        for batch, _ in dataset.take(steps_per_epoch):
            prob_distribution = model.predict(batch, verbose=0)
            probability_distributions.extend(prob_distribution)

        #np.save(f'D:/Sukrut/All Python/pythonProject1/CNN Models 3/{log_name}_probabilities.npy', np.array(probability_distributions))

        # Save the activity mappings
        with open(f'D:/Sukrut/All Python/pythonProject1/CNN Models 3/{log_name}_activity_mappings.txt', 'w') as f:
            for activity, index in activity_to_index.items():
                f.write(f'{activity},{index}\n')

        model.save(f'D:/Sukrut/All Python/pythonProject1/CNN Models 3/{log_name}.h5')

    # Main function to read logs, train models, and save them
    def main(training_logs_directory, max_sequence_length, batch_size, epochs):
        training_logs = [os.path.join(training_logs_directory, f) for f in os.listdir(training_logs_directory) if f.endswith('.xes')]
        total_logs = len(training_logs)
        logs_per_model = 4

        for i in range(0, total_logs, logs_per_model):
            batch_logs = training_logs[i:i + logs_per_model]
            combined_activities = read_and_combine_logs(batch_logs)
            print(f"Number of activities extracted: {len(combined_activities)}")
            encoded_training_activities, activity_to_index = encode_activities(combined_activities)
            log_name = f'{i // logs_per_model + 1}'
            train_and_save_cnn_model(log_name, encoded_training_activities, activity_to_index, max_sequence_length, epochs, batch_size)

    if __name__ == "__main__":
        training_logs_directory = 'D:/Siddhant/Masters Project/Dataset/Process Discovery Contest 2023_1_all/Training Logs 1' #keep changing it after completing
        main(training_logs_directory, max_sequence_length, batch_size, epochs)
