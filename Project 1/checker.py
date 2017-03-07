#!/usr/bin/env python

import zipfile
import shutil
import tempfile
import imp
import numbers
import numpy as np
import sys
import traceback

test_feature_matrix = np.array([[1,0,0],[0,1,1],[1,0,1],[0,1,0]])
test_labels = np.array([-1,1,-1,1])
test_theta = np.array([1,1,1])
test_theta_0 = 1
test_feature_vector = test_feature_matrix[0]
test_label = test_labels[0]

def check_zip(zipped_file, required_files, student_file, code_checks):
    """ Checks whether files in required_files are present in the zipped_file and basic code behavior """

    f = open(zipped_file, 'rb')
    z = zipfile.ZipFile(f)
    file_list = z.namelist()
    file_list = [n.split('/')[-1] for n in file_list]
    files_not_found = []
    for filename in required_files:
        if filename not in file_list:
            files_not_found.append(filename)

    if files_not_found:
        f.close()
        print('The following files are missing: {0}'.format(', '.join(files_not_found)))
        return False

    print('All required files present')

    # extract the zip to a temporary directory and check basic behavior
    tempdir = tempfile.mkdtemp()
    z.extractall(tempdir)
    try:
        sys.path[0] = tempdir
        student_module = imp.load_source('student_code', tempdir + '/project1/' + student_file)
    except Exception as e1:
        try:
            student_module = imp.load_source('student_code', tempdir + '/' + student_file)
        except Exception as e:
            shutil.rmtree(tempdir)
            f.close()
            print('Error importing your code:\n{}'.format(traceback.format_exc()))
            return False

    for check_fn in code_checks:
        check_fn(student_module)

    # delete the temporary directory
    shutil.rmtree(tempdir)
    f.close()

    return True

def _check_output(fn, args, types):
    check_name = fn.__name__
    try:
        res = fn(*args)
        if not isinstance(types, list):
            if not isinstance(res, types):
                print('{}: Expected a {} as output but got {}'.format(check_name, types.__name__, type(res).__name__))
                return False
        else:
            if not isinstance(res, tuple) or len(res) != len(types):
                print('{}: Expected a {}-tuple as output but got {}'.format(check_name, len(types), res))

            for i, item, expected_type in zip(range(len(types)), res, types):
                if not isinstance(item, expected_type):
                    print('{}: Expected a {} as output {} but got a {}'.format(check_name, types[i].__name__, i, type(item).__name__))
                    return False

        print('{}: Type checked'.format(check_name))
        return True
    except NotImplementedError:
        print('{}: Not implemened'.format(check_name))
    except Exception:
        print('{}: Exception encountered while running\n{}'.format(check_name, traceback.format_exc()))
    return False

def check_hinge_loss(student_module):
    try:
        res = student_module.hinge_loss(test_feature_matrix, test_labels, test_theta, test_theta_0)
        if isinstance(res, numbers.Number):
            print('hinge_loss: Implemented')
            return True
        else:
            print('hinge_loss: Expected a numeric return value but got: {0}'.format(res))
            return False
    except NotImplementedError:
        print('hinge_loss: Not implemented')
        return False
    except:
        print('hinge_loss: Exception in running hinge_loss')
        return False

def check_perceptron_single_step_update(student_module):
    args = test_feature_vector, test_label, test_theta, test_theta_0
    return _check_output(student_module.perceptron_single_step_update, args, [np.ndarray, numbers.Number])

def check_perceptron(student_module):
    args = test_feature_matrix, test_labels, 5
    return _check_output(student_module.perceptron, args, [np.ndarray, numbers.Number])

def check_pegasos_single_step_update(student_module):
    args = test_feature_vector, test_label, 1, 0.2, test_theta, test_theta_0
    return _check_output(student_module.pegasos_single_step_update, args, [np.ndarray, numbers.Number])

def check_average_perceptron(student_module):
    args = test_feature_matrix, test_labels, 5
    return _check_output(student_module.average_perceptron, args, [np.ndarray, numbers.Number])

def check_pegasos(student_module):
    args = test_feature_matrix, test_labels, 5, 2
    return _check_output(student_module.pegasos, args, [np.ndarray, numbers.Number])

def check_classify(student_module):
    args = test_feature_matrix, test_theta, test_theta_0
    return _check_output(student_module.classify, args, np.ndarray)

def check_perceptron_accuracy(student_module):
    args = test_feature_matrix, test_feature_matrix, test_labels, test_labels, 5
    return _check_output(student_module.perceptron_accuracy, args, [numbers.Number, numbers.Number])

def check_average_perceptron_accuracy(student_module):
    args = test_feature_matrix, test_feature_matrix, test_labels, test_labels, 5
    return _check_output(student_module.average_perceptron_accuracy, args, [numbers.Number, numbers.Number])

def check_pegasos_accuracy(student_module):
    args = test_feature_matrix, test_feature_matrix, test_labels, test_labels, 5, 2
    return _check_output(student_module.pegasos_accuracy, args, [numbers.Number, numbers.Number])

if __name__ == '__main__':
    zipped_file = 'project1.zip' # name of zip file to be submitted
    required_files = ['main.py', 'project1.py', 'reviews_submit.tsv', 'utils.py', 'writeup.pdf'] # required files in the zip
    student_file = 'project1.py' # name of student code file
    code_checks = [check_hinge_loss, check_perceptron_single_step_update, check_perceptron,
     check_pegasos_single_step_update, check_average_perceptron, check_pegasos,
     check_classify, check_perceptron_accuracy, check_pegasos_accuracy]
    check_zip(zipped_file, required_files, student_file, code_checks)
