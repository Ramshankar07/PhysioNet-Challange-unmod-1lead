#!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data


def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')

scored_classes = [
    "164890007", "164889003", "426627000", "426783006", "284470004",
    "427393009", "426177001", "427084000", "427172004"
]
equivalent_classes = {
    "63593006": "284470004",
    "17338001": "427172004"
}

def load_mat_labels(filename):
    header_file = filename.rsplit('.', 1)[0] + '.hea'
    with open(header_file, 'r') as f:
        for line in f:
            if line.startswith('# Dx'):
                return line.split(': ')[1].strip().split(',')
    return []

def preprocess_label(labels, scored_classes, equivalent_classes):
    y = np.zeros((len(scored_classes)), dtype=np.float32)
    for label in labels:
        if label in equivalent_classes:
            label = equivalent_classes[label]
        if label in scored_classes:
            y[scored_classes.index(label)] = 1
    return y

if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 4:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    model_input = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]

    # Find files.
    input_files = []
    # for f in os.listdir(input_directory):
        # if os.path.isfile(os.path.join(input_directory, f)):
            # cpsc_dir = os.path.join(input_directory, "cpsc_2018")
            # if not cpsc_dir.lower().startswith('.') and cpsc_dir.lower().endswith('.mat'):
                
            #     if os.path.isdir(cpsc_dir):
            #         hea_file = os.path.join(cpsc_dir, f.rsplit('.', 1)[0] + '.hea')
            #         if os.path.isfile(hea_file):
            #             labels = load_mat_labels(hea_file)
            #             preprocessed_labels = preprocess_label(labels, scored_classes, equivalent_classes)
            #             if np.any(preprocessed_labels):  # Check if any labels are present
            #                 input_files.append((f, preprocessed_labels))
    
    cpsc_dir = os.path.join(input_directory, "cpsc_2018")
    for f in os.listdir(cpsc_dir):
        # print(f)
        if f.endswith('.mat'):
            # print(f)
                
            hea_file = os.path.join(cpsc_dir, f.rsplit('.', 1)[0] + '.hea')
            if os.path.isfile(hea_file): 
                # print("yes")
                labels = load_mat_labels(hea_file)
                preprocessed_labels = preprocess_label(labels, scored_classes, equivalent_classes)
                if np.any(preprocessed_labels):  # Check if any labels are present
                    input_files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading 12ECG model...')
    model = load_12ECG_model(model_input)

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)
    print(num_files)
    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = os.path.join(cpsc_dir,f)
        data,header_data = load_challenge_data(tmp_input_file)
        current_label, current_score,classes = run_12ECG_classifier(data,header_data, model)
        # Save results.
        print('going to save')
        save_challenge_predictions(output_directory,f,current_score,current_label,classes)


    print('Done.')
