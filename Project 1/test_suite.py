import project1 as p1
import numpy as np
import numbers
import traceback
from testcases.hinge_loss_test_data import hingeLossTestData
from testcases.perceptron_test_data import perceptronTestData
from testcases.pegasos_test_data import pegasosTestData

def compareTestOutput(expectedOutput, actualOutput):
    # Compares expected and actual output. Allows for small margin of error of 1e-8.
    if np.allclose(expectedOutput, actualOutput):
        print("Correct: Output matches expected")
        return True
    else:
        print("Incorrect: Test Failed")
        print("Expected Output: " + str(expectedOutput))
        print("Actual Output: " + str(actualOutput))
    return False
    

def testHingeLoss(testData):
    # Test hinge loss method with dictionary containing case input/output data.
    feature_matrix = testData["featureMatrix"]
    labels = testData["labels"]
    theta = testData["theta"]
    theta0 = testData["theta0"]
    expectedOutput = testData["expectedOutput"]
    actualOutput = None
    try:
        actualOutput = p1.hinge_loss(feature_matrix, labels, theta, theta0)
        # Check output format and return error if incorrect.
        if isinstance(actualOutput, numbers.Number) == False:
            print('Incorrect Ouput Type: Expected a numeric return value but got: {0}'.format(actualOutput))
    except NotImplementedError:
        # Not implemented case.
        print("Hinge Loss has not yet been implemented.")
        return False
    except:
        print('Exception while running hinge loss.')
        print(traceback.format_exc())
        return False
        
    return compareTestOutput(expectedOutput, actualOutput)

def testPerceptronSingleStepUpdate(testData):
    feature_vector = testData["feature_vector"]
    label = testData['label']
    theta, theta0 = testData['theta']
    expectedOutput = testData['expected_output']
    actualOutput = None
    try:
        actualOutput = p1.perceptron_single_step_update(feature_vector, label, theta, theta0)
        if (not isinstance(actualOutput[0], np.ndarray) or not isinstance(actualOutput[1], numbers.Number)):
            print('Incorrect Output Type: Expected a tuple of numpy array and numeric return value but got: {0}'.format(actualOutput))
    except NotImplementedError:
        print("Perceptron Single Step Update has not yet been implemented.")
        return False
    except:
        print('Exception while running perceptron_single_step_update.')
        print(traceback.format_exc())
        return False

    return (compareTestOutput(expectedOutput[0], actualOutput[0]) and compareTestOutput(expectedOutput[1], actualOutput[1]))
    
def testPegasosSingleStepUpdate(testData):
    feature_vector = testData["feature_vector"]
    label = testData['label']
    L = testData['L']
    eta = testData['eta']
    theta, theta0 = testData['theta']
    expectedOutput = testData['expected_output']
    actualOutput = None
    try:
        actualOutput = p1.pegasos_single_step_update(feature_vector, label, L, eta, theta, theta0)
        if (not isinstance(actualOutput[0], np.ndarray) or not isinstance(actualOutput[1], numbers.Number)):
            print('Incorrect Output Type: Expected a tuple of numpy array and numeric return value but got: {0}'.format(actualOutput))
    except NotImplementedError:
        print("Pegasos Single Step Update has not yet been implemented.")
        return False
    except:
        print('Exception while running pegasos_single_step_update.')
        print(traceback.format_exc())
        return False

    return (compareTestOutput(expectedOutput[0], actualOutput[0]) and compareTestOutput(expectedOutput[1], actualOutput[1]))

print("Hinge Loss Tests 1-3")
for i in range(3): 
    print("Test # " + str(i+1))
    data = hingeLossTestData[i]
    testHingeLoss(data)

print("\nPerceptron Single Step Update Tests 4-6")
for i in range(3): 
    print("Test # " + str(i+4))
    data = perceptronTestData[i]
    testPerceptronSingleStepUpdate(data)

print("\nPegasos Single Step Update Tests 7-10")
for i in range(4): 
    print("Test # " + str(i+7))
    data = pegasosTestData[i]
    testPegasosSingleStepUpdate(data)

