import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting graphs
import pandas as pd
import numpy as np
import csv


def draw_graph_distance_inverse_VS_duration():

    full_dS = pd.read_csv('train_cleaned_new_full.csv')
    msk = np.random.rand(len(full_dS)) < .75
    Distance = full_dS[msk]["distance_to_cover"].values
    duration = full_dS[msk]["trip_duration"].values
    time_of_Day = full_dS[msk]["time_of_day"].values
    type_of_day = full_dS[msk]["type_of_day"].values
    pickup_zip = full_dS[msk]["pickup_zip"].values
    dist_over_duration = []  # miles/sec
    for i in range(len(Distance)):
        print(Distance[i]/duration[i])
        dist_over_duration.append(duration[i]/Distance[i])


    column_header = ['distance','time_of_day','type_of_day','pick_up_zip','speed']
    with open("speedVSattributes.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['distance','time_of_day','type_of_day','pick_up_zip','speed'])
        writer.writeheader()
        size = len(Distance)
        for row_ind in range(size):
            writer.writerow({column_header[0]: float(Distance[row_ind]), column_header[1]: time_of_Day[row_ind]
                                , column_header[2]: type_of_day[row_ind], column_header[3]: pickup_zip[row_ind],
                             column_header[4]: float(dist_over_duration[row_ind])})

    '''plt.plot(Distance, dist_over_duration, color='darkorange',
             label='Distance V/S Distance/Duration')
    plt.xlabel("Distance(miles)")
    plt.ylabel("Distance/Duration (miles/Seconds)")
    plt.title("Depth of Trees for Random Forest V/S Mean Squared Error")
    plt.legend()
    plt.show()'''

draw_graph_distance_inverse_VS_duration()