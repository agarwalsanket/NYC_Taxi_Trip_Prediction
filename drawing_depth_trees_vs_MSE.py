import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting graphs
"""
This method is used to plot graph between depth of Random forest regressor tree and the MSE

@author: Sanket Agarwal
"""

def draw_graph_depth_mse():

    # Mean squared error calculated from the NYC_taxi_trip_prediction.py file
    # But hardcoded here as tkinter and matplotlib dont go together
    mean_squared_error_l = [186022.97591829329, 152646.79131978776, 136104.48891890259, 123820.18133570212, 116504.75077115586, 111392.71552277566, 107329.38976243655, 104891.08936603311, 103415.31102691819, 102761.72315024534]
    depth_trees = [2,4,6,8,10,12,14,16,18,20]
    plt.plot(depth_trees, mean_squared_error_l, color='darkorange',
             label='Effect of depth of trees on MSE for Random Forest')
    plt.xlabel("Depth of Trees")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Depth of Trees for Random Forest V/S Mean Squared Error")
    plt.legend()
    plt.show()

draw_graph_depth_mse()