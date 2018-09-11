import cnn
import cnn_data as data

import numpy as np
import pickle

PARAM_FILENAME = "saved_params.p"
TEST_DATA_NUM = 10000


if __name__ == '__main__':
    
    [params, num_correct, avg_loss] = pickle.load(open(PARAM_FILENAME, 'rb'))
    [kernel, conv_bias, fc_weights, fc_bias] = params
    
    data_dicts = data.get_data()
    np.random.shuffle(data_dicts)
    test_data = data_dicts[:TEST_DATA_NUM]
    
    num_correct = 0
    num_classes = data.get_num_classes()
    class_count = [0 for i in xrange(num_classes)]
    class_correct = [0 for i in xrange(num_classes)]
   
    print("Computing accuracy over test set:")

    for i in xrange(TEST_DATA_NUM):
        print("Processing example " + str(i))
        img = test_data[i]['image']
        label = test_data[i]['label']
        prediction, probs, _ = cnn.predict_single_example(img, label, num_classes, kernel, conv_bias, fc_weights, fc_bias)
        class_count[label] += 1
        if prediction == label:
            num_correct += 1
            class_correct[prediction] += 1
    print("Predicted {} out of {} examples correctly.".format(num_correct, TEST_DATA_NUM))
    print("Overall Accuracy: {}%".format(float(num_correct) * 100 / TEST_DATA_NUM))
    print("Examples and correct predictions by class : ")
    print(class_count)
    print(class_correct)