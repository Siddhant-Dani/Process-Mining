import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

with open('output_test.txt', 'w') as f:
    sys.stdout = f
    # Parameters
    max_sequence_length = 32

    # Extracting concept names
    def extract_concept_names(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        concept_names = []
        for trace in root.iter('trace'):
            trace_concept_names = []
            for event in trace.iter('event'):
                for string in event.iter('string'):
                    if string.attrib.get('key') == 'concept:name':
                        trace_concept_names.append(string.attrib['value'])
            trace_concept_names.append('<END>')
            concept_names.append(trace_concept_names)
        return concept_names

    # Function to read logs
    def read_log(file_path):
        return extract_concept_names(file_path)

    # Encode activities
    def encode_activities(activities, activity_to_index):
        encoded_activities = []
        for activity in activities:
            if activity in activity_to_index:
                encoded_activities.append(activity_to_index[activity])
            else:
                print(f"Activity '{activity}' not found in activity_to_index mapping.")
        return encoded_activities

    # Load activity mappings
    def load_activity_mappings(mapping_file):
        activity_to_index = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                activity, index = line.strip().split(',')
                activity_to_index[activity] = int(index)
        return activity_to_index

    # Prepare features from aggregated probabilities
    def prepare_features(probabilities):
        mean_prob = np.mean(probabilities, axis=1)
        var_prob = np.var(probabilities, axis=1)
        return np.column_stack([mean_prob, var_prob])

    # Extract probabilities
    def extract_probabilities(model, sequences, max_sequence_length):
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
        print(f"Padded sequences for prediction: {padded_sequences}")
        probabilities = model.predict(padded_sequences)
        print(f"Extracted probabilities: {probabilities}")
        return probabilities

    # Train classifier
    def train_classifier(features, labels):
        clf = RandomForestClassifier()
        clf.fit(features, labels)
        return clf

    # Classify traces
    def classify_traces(test_log, base_log, model, max_sequence_length, num_classes, activity_to_index):
        def encode_and_pad(log):
            sequences = []
            for trace in log:
                encoded_trace = encode_activities(trace, activity_to_index)
                print(f"Encoded trace: {encoded_trace}")
                if isinstance(encoded_trace, list):
                    padded_trace = pad_sequences([encoded_trace], maxlen=max_sequence_length, padding='post')
                    print(f"Padded trace: {padded_trace}")
                    sequences.append(padded_trace[0])
                else:
                    print(f"Unexpected format for encoded trace: {encoded_trace}")
            return np.array(sequences)

        print("Encoding and padding test sequences")
        test_sequences = encode_and_pad(test_log)
        print(f"Encoded and padded test sequences: {test_sequences}")

        print("Encoding and padding base sequences")
        base_sequences = encode_and_pad(base_log)
        print(f"Encoded and padded base sequences: {base_sequences}")

        if test_sequences.size == 0 or base_sequences.size == 0:
            raise ValueError("Encoded sequences are empty. Please check the encoding process and input logs.")

        test_probabilities = []
        base_probabilities = []

        for test_seq, base_seq in zip(test_sequences, base_sequences):
            test_prob = extract_probabilities(model, [test_seq], max_sequence_length)
            test_probabilities.append(test_prob[0])
            base_prob = extract_probabilities(model, [base_seq], max_sequence_length)
            base_probabilities.append(base_prob[0])

        print(f"Test probabilities: {test_probabilities}")
        print(f"Base probabilities: {base_probabilities}")

        test_features = prepare_features(np.array(test_probabilities))
        base_features = prepare_features(np.array(base_probabilities))

        print(f"Test features: {test_features}")
        print(f"Base features: {base_features}")

        ground_truth_log = []
        test_similarities = []
        base_similarities = []

        features = np.vstack([test_features, base_features])
        labels = np.array([1] * len(test_features) + [0] * len(base_features))
        clf = train_classifier(features, labels)

        for test_feature, base_feature in zip(test_features, base_features):
            test_similarity = clf.predict_proba([test_feature])[0][1]
            base_similarity = clf.predict_proba([base_feature])[0][1]

            ground_truth_log.append('positive' if test_similarity > base_similarity else 'negative')
            test_similarities.append(test_similarity)
            base_similarities.append(base_similarity)

            print(f"Trace Test Similarity = {test_similarity:.4f}, Base Similarity = {base_similarity:.4f}")

        return ground_truth_log, test_similarities, base_similarities

    # Save ground truth log as .xes file
    def save_ground_truth_log_as_xes(ground_truth_log, file_path, test_log):
        root = ET.Element('log')
        for trace, classification in zip(test_log, ground_truth_log):
            trace_elem = ET.SubElement(root, 'trace')
            for event in trace:
                event_elem = ET.SubElement(trace_elem, 'event')
                name_elem = ET.SubElement(event_elem, 'string', key='concept:name', value=event)
                is_pos_elem = ET.SubElement(event_elem, 'boolean', key='pdc:isPos',
                                            value='true' if classification == 'positive' else 'false')
        tree = ET.ElementTree(root)
        tree.write(file_path)

    # Compute accuracy rates and F-score
    def compute_accuracy_rates_and_fscore(test_similarities, base_similarities):
        TP = sum([1 for sim in test_similarities if sim >= 0.5])
        FN = sum([1 for sim in test_similarities if sim < 0.5])
        TN = sum([1 for sim in base_similarities if sim < 0.5])
        FP = sum([1 for sim in base_similarities if sim >= 0.5])

        P = TP / (TP + FP) if (TP + FP) > 0 else 0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0
        F = 2 * P * R / (P + R) if (P + R) > 0 else 0

        return P, R, F

    # Main function to classify logs and compute accuracy rates
    def main(models_directory, test_logs_directory, base_logs_directory, mapping_files_directory,
             probabilities_files_directory, max_sequence_length):
        model_files = [f for f in os.listdir(models_directory) if f.endswith('.h5')]
        test_log_files = [f for f in os.listdir(test_logs_directory) if f.endswith('.xes')]
        base_log_files = [f for f in os.listdir(base_logs_directory) if f.endswith('.xes')]
        mapping_files = [f for f in os.listdir(mapping_files_directory) if f.endswith('.txt')]
        probability_files = [f for f in os.listdir(probabilities_files_directory) if f.endswith('.npy')]

        # Check for equal number of files
        if not (len(model_files) == len(test_log_files) == len(base_log_files) == len(mapping_files) == len(probability_files)):
            raise ValueError("The number of model files, test logs, base logs, mapping files, and probability files must be equal.")

        total_fscore = 0
        num_logs = len(model_files)

        for i in range(num_logs):
            model_file = model_files[i]
            test_log_file = test_log_files[i]
            base_log_file = base_log_files[i]
            mapping_file = mapping_files[i]
            prob_file = probability_files[i]

            print(f"Processing model: {model_file}, test log: {test_log_file}, base log: {base_log_file}")

            model_path = os.path.join(models_directory, model_file)
            test_log_path = os.path.join(test_logs_directory, test_log_file)
            base_log_path = os.path.join(base_logs_directory, base_log_file)
            mapping_file_path = os.path.join(mapping_files_directory, mapping_file)
            prob_file_path = os.path.join(probabilities_files_directory, prob_file)

            if not (os.path.exists(test_log_path) and os.path.exists(base_log_path)):
                print(f"Missing files for model: {model_file}")
                continue

            model = load_model(model_path)
            activity_to_index = load_activity_mappings(mapping_file_path)

            print(f"Activity to Index Mapping: {activity_to_index}")

            test_log = read_log(test_log_path)
            base_log = read_log(base_log_path)

            def filter_existing_indices(log):
                return [[activity for activity in trace if activity in activity_to_index] for trace in log]

            encoded_test_activities = filter_existing_indices(test_log)
            encoded_base_activities = filter_existing_indices(base_log)

            print(f"Encoded Test Activities: {encoded_test_activities}")
            print(f"Encoded Base Activities: {encoded_base_activities}")

            ground_truth_log, test_log_similarities, base_log_similarities = classify_traces(
                encoded_test_activities, encoded_base_activities, model, max_sequence_length,
                len(activity_to_index), activity_to_index)

            ground_truth_log_path = f"{os.path.splitext(test_log_path)[0]}_ground_truth.xes"
            save_ground_truth_log_as_xes(ground_truth_log, ground_truth_log_path, test_log)

            for j, (test_similarity, base_similarity) in enumerate(zip(test_log_similarities, base_log_similarities)):
                print(f"Trace {j + 1}: Test Similarity = {test_similarity:.4f}, Base Similarity = {base_similarity:.4f}")

            P, R, F = compute_accuracy_rates_and_fscore(test_log_similarities, base_log_similarities)
            total_fscore += F

        average_fscore = total_fscore / num_logs
        print(f"Average F-score: {average_fscore:.4f}")

    if __name__ == "__main__":
        models_directory = "D:/Sukrut/All Python/pythonProject1/CNN Models 3"
        test_logs_directory = "D:/Siddhant/Masters Project/Dataset/Process Discovery Contest 2023_1_all/Test Logs"
        base_logs_directory = "D:/Siddhant/Masters Project/Dataset/Process Discovery Contest 2023_1_all/Base Logs"
        mapping_files_directory = "D:/Sukrut/All Python/pythonProject1/Mappings"
        probabilities_files_directory = "D:/Sukrut/All Python/pythonProject1/Probabilities"
        max_sequence_length = 32

        main(models_directory, test_logs_directory, base_logs_directory, mapping_files_directory, probabilities_files_directory, max_sequence_length)
