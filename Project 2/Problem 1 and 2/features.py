import numpy as np
import matplotlib.pyplot as plt

# For all these functions, X = n x d numpy array containing the training data
# where each row of X = one sample and each column of X = one feature.

### Functions for you to fill in ###

# Given principal component vectors produced using
# pcs = principalComponents(X), 
# this function returns a new data array in which each sample in X 
# has been projected onto the first n_components principcal components.
def projectOntoPC(X, pcs, n_components):
    # TODO: first center data using the centerData() function.
    # Centering data
    Xc = centerData(X)
    # TODO: Return the projection of the centered dataset 
    #       on the first n_components principal components.
    #       This should be an array with dimensions: n x n_components.
    Xp = np.dot(Xc,pcs[:,:n_components])
    # Hint: these principal components = first n_components columns 
    #       of the eigenvectors returned by PrincipalComponents().
    #       Note that each eigenvector is already be a unit-vector,
    #       so the projection may be done using matrix multiplication.
    return Xp


# Returns a new dataset with features given by the mapping 
# which corresponds to the quadratic kernel.
def cubicFeatures(X):
    n, d = X.shape
    X_withones = np.ones((n,d+1))
    X_withones[:,:-1] = X
    new_d = int((d+1)*(d+2)*(d+3)/6)
    
    newData = np.zeros((n, new_d))
    # TODO: Fill in matrix newData with the correct values given by mapping 
    #       each original sample into the feature space of the cubic kernel.
    #       Note that newData should have the same number of rows as X, where each
    #       row corresponds to a sample, and the dimensionality of newData has 
    #       already been set to the appropriate value for the cubic kernel feature mapping.
    
    # for each point
    for p in range(0, n):
        count = 1 # start at 1 for 1 term at 0
        feat = np.ones((new_d))
        for i in range(0, d):
            feat[count] = np.power(X[p,i],3)
            count += 1
            feat[count] = np.sqrt(3)*np.power(X[p,i],2)
            count += 1
            feat[count] = np.sqrt(3)*np.power(X[p,i],1)
            count += 1
        for i in range(0, d):
            for j in range(0, d):
                if i != j:
                    feat[count] = np.sqrt(3)*np.power(X[p,i],2)*X[p,j]
                    count += 1
        for i in range(1, d):
            for j in range(0, i):
                feat[count] = np.sqrt(6)*X[p,i]*X[p,j]
                count += 1
        for i in range(2, d):
            for j in range(0, i):
                for k in range(0, j):
                    feat[count] = np.sqrt(6)*X[p,i]*X[p,j]*X[p,k]
                    count += 1
        newData[p,:] = feat
    return newData



### Functions which are already complete, for you to use ###

# Returns a centered version of the data,
# where each feature now has mean = 0
def centerData(X):
    featureMeans = X.mean(axis = 0)
    return(X - featureMeans)


# Returns the principal component vectors of the data,
# sorted in decreasing order of eigenvalue magnitude.
def principalComponents(X):
    centeredData = centerData(X) # first center data
    scatterMatrix = np.dot(centeredData.transpose(), centeredData)
    eigenValues,eigenVectors = np.linalg.eig(scatterMatrix)
    # Re-order eigenvectors by eigenvalue magnitude: 
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenVectors


# Given the principal component vectors as the columns of matrix pcs,  
# this function projects each sample in X onto the first two principal components
# and produces a scatterplot where points are marked with the digit depicted in the corresponding image.
# labels = a numpy array containing the digits corresponding to each image in X.
def plotPC(X, pcs, labels):
    pc_data = projectOntoPC(X, pcs, n_components = 2)
    text_labels = [str(z) for z in labels.tolist()]
    fig, ax = plt.subplots()
    ax.scatter(pc_data[:,0],pc_data[:,1], alpha=0, marker = ".")
    for i, txt in enumerate(text_labels):
        ax.annotate(txt, (pc_data[i,0],pc_data[i,1]))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.show(block=True)


# Given the principal component vectors as the columns of matrix pcs,  
# this function reconstructs a single image 
# from its principal component representation, x_pca. 
# X = the original data to which PCA was applied to get pcs.
def reconstructPC(x_pca, pcs, n_components, X):
    featureMeans = X - centerData(X)
    featureMeans = featureMeans[0,:]
    x_reconstructed = np.dot(x_pca, pcs[:,range(n_components)].T) + featureMeans
    return x_reconstructed
