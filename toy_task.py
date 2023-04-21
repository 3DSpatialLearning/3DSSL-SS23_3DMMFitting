# import numpy as np
import numpy as np
from utils.visualize import display_points
from utils.fit import fit_data

if __name__ == '__main__':
    # load the data
    points = np.load("./data/team1_points.npy")
    landmarks = np.load("./data/team1_landmarks.npy")
    normals = np.load("./data/team1_normals.npy")

    # uncomment to display input data
    # display_points("Input Data", points=points, landmarks=landmarks)

    landmarks_optimized = fit_data(
        points=points, landmarks=landmarks, normals=normals,
        steps=2000, lr=0.01)

    # uncomment to display optimized data
    display_points("Fitted", landmarks=landmarks_optimized)
    