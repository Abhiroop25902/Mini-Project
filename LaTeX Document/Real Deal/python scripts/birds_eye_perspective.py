def birds_eye_view(corner_points,width,height,image):
    """
    Compute the transformation matrix
    corner_points : 4 corner points selected from the image
    height, width : size of the image
    return : transformation matrix and the transformed image
    """
    # Create an array out of the 4 corner points
    corner_points = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    img_params = np.float32([[0,0],[width,0],[0,height],[width,height]])
    # Compute and return the transformation matrix
    matrix = cv.getPerspectiveTransform(corner_points,img_params)
    img_transformed = cv.warpPerspective(image,matrix,(width,height))
    
    return matrix,img_transformed