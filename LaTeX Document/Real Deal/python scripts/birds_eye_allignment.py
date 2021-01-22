def birds_eye_point(matrix,centroids):
    """ Apply the perspective transformation to every ground point which have been detected on the main frame.
    @ matrix : the 3x3 matrix
    @ centroids : list that contains the points to transform
    return : list containing all the new points
    """
    # Compute the new coordinates of our points
    points = np.float32(centroids).reshape(-1, 1, 2)
    transformed_points = cv.perspectiveTransform(points, matrix)
    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()

    for i in range(0,transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
        
    return transformed_points_list