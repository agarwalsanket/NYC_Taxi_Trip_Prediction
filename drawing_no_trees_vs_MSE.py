import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting graphs
def draw_graph_trees_mse():

    # Mean squared error calculated from the NYC_taxi_trip_prediction.py file
    # But hardcoded here as tkinter and matplotlib dont go together
    mean_squared_error_l = [120315.04038200337, 119420.00668805688, 118936.04519550462, 119134.9382182608,
                            118956.25756358843, 119079.13405434613, 118909.22463650003, 118848.82959805393,
                            118897.06532003591, 118800.48281405406]
    no_trees = [5,10,15,20,25,30,35,40,45,50]
    plt.plot(no_trees, mean_squared_error_l, color='darkorange',
             label='Effect of number of trees on MSE for Random Forest')
    plt.xlabel("Number of Trees")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Number of Trees for Random Forest V/S Mean Squared Error")
    plt.legend()
    plt.show()

draw_graph_trees_mse()