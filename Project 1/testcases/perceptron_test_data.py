import numpy as np

#case 1
vector_1 = np.array([1,2,2,0,-1,-1,9])
theta_1 = np.array([3,4,4,-1,-10,5,6])
theta0_1 = 3
y_1 = -1
new_theta_1 = np.array([2, 2, 2,  -1,  -9, 6,  -3])
new_theta0_1 = 2

testData1={'feature_vector': vector_1, "theta": (theta_1, theta0_1), "label":y_1, "expected_output":(new_theta_1,new_theta0_1)}

#case 2
vector_2 = np.array([0,0,-100, -1000, 0.6])
theta_2 = np.array([5,5,5,5,209.4])
theta0_2 = -42
y_2 = 1
new_theta_2 = np.array([   5.,    5.,  -95., -995.,  210.])
new_theta0_2 = -41

testData2={'feature_vector': vector_2, "theta": (theta_2, theta0_2), "label":y_2, "expected_output":(new_theta_2,new_theta0_2)}

#case 3
vector_3 = np.array(range(10))
theta_3 = np.ones(10)
theta0_3 = 17
y_3 = -1
new_theta_3 = np.array([ 1.,  0., -1., -2., -3., -4., -5., -6., -7., -8.])
new_theta0_3 = 16

testData3={'feature_vector': vector_3, "theta": (theta_3, theta0_3), "label":y_3, "expected_output":(new_theta_3,new_theta0_3)}

perceptronTestData = [testData1, testData2, testData3]