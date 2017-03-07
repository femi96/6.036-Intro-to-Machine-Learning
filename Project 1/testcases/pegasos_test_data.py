import numpy as np

feature_vectors = np.array([[1,0,0],[0,1,0],[0,1,1],[1,1,0]])

labels = [1,-1,-1,-1]

Ls = [1, 1, 2, 2]
etas = [0.3, 0.2, 0.1, 0.08]


initial_thetas = [
	np.array([1,1,1]),
	np.array([0,0,0]),
	np.array([1,0,1]),
	np.array([1,1,0])
]

initial_theta_0s = [1, 0, 0, -1]

new_thetas = [
	np.array([ 0.7,  0.7,  0.7]),
	np.array([ 0.,  -0.2,  0. ]),
	np.array([ 0.8, -0.1,  0.7]),
	np.array([ 0.76,  0.76,  0.  ])
]

new_theta_0s = [0.7, -0.2, -0.1, -0.92]

pegasosTestData = []
for i in range(len(feature_vectors)):
	testData = {
		"feature_vector": feature_vectors[i], 
		"label": labels[i], 
		"L": Ls[i], 
      "eta" : etas[i],
		"theta": (initial_thetas[i], initial_theta_0s[i]), 
		"expected_output": (new_thetas[i], new_theta_0s[i])
	}
	pegasosTestData.append(testData)