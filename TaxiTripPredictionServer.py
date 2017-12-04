
# Importing the library and packages used
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from math import sin, cos, sqrt, atan2, radians
from datetime import datetime
from datetime import date
from datetime import time
from uszipcode import ZipcodeSearchEngine
from geopy.geocoders import Nominatim
import pickle # Importing Pickle library to store dictionary object
from flask import Flask, redirect, url_for, request,render_template
app = Flask(__name__)

"""
@author:Mayank Pandey
@author:Sanket Agarwal

Python code for the NYC Trip duration prediction project.
This implementation uses the cleaned and prepared data of the NYC Yellow cab trip details.
It first uses the cleaned data to train the model using Random Forest Regression modelling technique.
We have used scikit learn RandomForestRegressor modelling package for the modelling purpose.
We have also used scikit learn DecisionTreeRegressor modelling package for experimenting and comparing the results.

Modst portion of the code is commented out, since they are the code either used for training the model and then saving
model as pickel object or experimentation purpose. They were commented as training the model with data having
records over 1 million takes a lot of time (30 - 40 minutes).
So only relevant code for prediction is kept un-commented
"""


class trip_duration_prediction():
    """
    This is the main class having all the methods and functionalities for thr prediction purpose.
    """

    def distance_to_cover(self,pickup_point, drop_off_point):
        """
        This method calculates the haversine distance (crow-fly distance) for two points
        :param pickup_point: Pickup lat and long
        :param drop_off_point: drop off lat and long
        :return: haversine distance
        """
        radius_earth = 6373.0  # radius of earth
        latitude_pick = radians(float(pickup_point[0]))
        longitude_pick = radians(float(pickup_point[1]))
        latitude_drop = radians(float(drop_off_point[0]))
        longitude_drop = radians(float(drop_off_point[1]))
        differ_lon = longitude_drop - longitude_pick
        differ_lat = latitude_drop - latitude_pick
        a = (sin(differ_lat / 2)) ** 2 + cos(latitude_pick) * cos(latitude_drop) * (sin(differ_lon / 2)) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = radius_earth * c * 0.62137  # in miles
        return distance

    def predicting_trip_duration(self, pickup, drop,date_):
        """
        This is the main method which is doing all the modelling
        and calculations needed for the prediction of the trip duration
        :param pickup: Pick up address from the user
        :param drop: Drop off address from the user
        :param date_: Date and time from the user
        :return: The predicted time or message for the user
        """
        geo_locator = Nominatim()
        date_time_info = str(date_)
        if len(date_time_info) == 0:
            return "Enter trip details"
        dt = date_time_info.split(" ")[0].split("/")
        year, month, day = int(dt[2]), int(dt[1]), int(dt[0])
        date_entered = date(year, month, day)

        day_of_week = date_entered.isoweekday()
        if day_of_week == 6 or day_of_week == 7:
            type_of_day = 1  # weekend
        else:
            type_of_day = 0  # weekday

        t = [float(x) for x in date_time_info.split(" ")[1].split(":")]

        pickup_time = time(int(t[0]), int(t[1]), 0, 0, None)
        time_of_day = None  # divided day into 24 hours

        if pickup_time >= time(5, 0, 0, 0, None) and pickup_time <= time(5, 59, 0, 0, None):
            time_of_day = 0
        elif pickup_time >= time(6, 0, 0, 0, None) and pickup_time <= time(6, 59, 0, 0, None):
            time_of_day = 1
        elif pickup_time >= time(7, 0, 0, 0, None) and pickup_time <= time(7, 59, 0, 0, None):
            time_of_day = 2
        elif pickup_time >= time(8, 0, 0, 0, None) and pickup_time <= time(8, 59, 0, 0, None):
            time_of_day = 3
        elif pickup_time >= time(9, 0, 0, 0, None) and pickup_time <= time(9, 59, 0, 0, None):
            time_of_day = 4
        elif pickup_time >= time(10, 0, 0, 0, None) and pickup_time <= time(10, 59, 0, 0, None):
            time_of_day = 5
        elif pickup_time >= time(11, 0, 0, 0, None) and pickup_time <= time(11, 59, 0, 0, None):
            time_of_day = 6
        elif pickup_time >= time(12, 0, 0, 0, None) and pickup_time <= time(12, 59, 0, 0, None):
            time_of_day = 7
        elif pickup_time >= time(13, 0, 0, 0, None) and pickup_time <= time(13, 59, 0, 0, None):
            time_of_day = 8
        elif pickup_time >= time(14, 0, 0, 0, None) and pickup_time <= time(14, 59, 0, 0, None):
            time_of_day = 9
        elif pickup_time >= time(15, 0, 0, 0, None) and pickup_time <= time(15, 59, 0, 0, None):
            time_of_day = 10
        elif pickup_time >= time(16, 0, 0, 0, None) and pickup_time <= time(16, 59, 0, 0, None):
            time_of_day = 11
        elif pickup_time >= time(17, 0, 0, 0, None) and pickup_time <= time(17, 59, 0, 0, None):
            time_of_day = 12
        elif pickup_time >= time(18, 0, 0, 0, None) and pickup_time <= time(18, 59, 0, 0, None):
            time_of_day = 13
        elif pickup_time >= time(19, 0, 0, 0, None) and pickup_time <= time(19, 59, 0, 0, None):
            time_of_day = 14
        elif pickup_time >= time(20, 0, 0, 0, None) and pickup_time <= time(20, 59, 0, 0, None):
            time_of_day = 15
        elif pickup_time >= time(21, 0, 0, 0, None) and pickup_time <= time(21, 59, 0, 0, None):
            time_of_day = 16
        elif pickup_time >= time(22, 0, 0, 0, None) and pickup_time <= time(22, 59, 0, 0, None):
            time_of_day = 17
        elif pickup_time >= time(23, 0, 0, 0, None) and pickup_time <= time(23, 59, 0, 0, None):
            time_of_day = 18
        elif pickup_time >= time(1, 0, 0, 0, None) and pickup_time <= time(1, 59, 0, 0, None):
            time_of_day = 19
        elif pickup_time >= time(2, 0, 0, 0, None) and pickup_time <= time(2, 59, 0, 0, None):
            time_of_day = 20
        elif pickup_time >= time(3, 0, 0, 0, None) and pickup_time <= time(3, 59, 0, 0, None):
            time_of_day = 21
        elif pickup_time >= time(4, 0, 0, 0, None) and pickup_time <= time(4, 59, 0, 0, None):
            time_of_day = 22
        else:
            time_of_day = 23

        pickup_point_address = pickup  # if both pick and drop addresses are entered same
        drop_off_address = drop
        if pickup_point_address == drop_off_address:
            return "The trip duration will be 0 minutes"

        location_address_pick = geo_locator.geocode(pickup_point_address)
        pick_lat = None
        pick_long = None
        drop_lat = None
        drop_long = None
        if location_address_pick is not None:
            pick_lat = location_address_pick.latitude
            pick_long = location_address_pick.longitude
            coordinates_pick = (pick_lat, pick_long)
        else:   # If pick address entered is wrong
            return "Enter proper pick up address"

        location_address_drop_off = geo_locator.geocode(drop_off_address)
        if location_address_drop_off is not None:
            drop_lat = location_address_drop_off.latitude
            drop_long = location_address_drop_off.longitude
            coordinates_drop = (drop_lat,drop_long)
        else:   # If drop address entered is wrong
            return "Enter proper drop off address"

        if coordinates_pick == coordinates_drop:
            return "The trip duration will be 0 minutes"

        search = ZipcodeSearchEngine()  # finding the zip code using lat and long/ using google api
        pickup_zip = search.by_coordinate(pick_lat, pick_long, returns=1)[0]['Zipcode']
        drop_off_zip = search.by_coordinate(drop_lat, drop_long, returns=1)[0]['Zipcode']

        distance = self.distance_to_cover(coordinates_pick, coordinates_drop)

        table = [month,type_of_day,time_of_day, pickup_zip,drop_off_zip,pick_lat,pick_long,drop_lat,drop_long,distance]

        # making dataframe ready for prediction/ Feeding in user entered data
        testTable = [[int(month),int(type_of_day),int(time_of_day), int(pickup_zip),int(drop_off_zip),float(pick_lat),float(pick_long),float(drop_lat),float(drop_long), float(distance), '']]

        # cols will store names of attributes in the dataset
        cols = ['pickup_month','type_of_day', 'time_of_day','pickup_zip', 'dropoff_zip','pick_lat','pick_long','drop_lat','drop_long','distance_to_cover','trip_duration']
        df = pd.DataFrame(testTable, columns=cols)

        # FullDS will store the dataset provided on the given path
        full_dS = pd.read_csv('train_cleaned_new_full_with_per_hr.csv')
        train = full_dS

        # tesing file will be the dataframe of details given by the user
        test_ds = df

        # Feature will have the list of features for modelling
        # Fragmenting the data into two parts: training set and validation set
        msk = np.random.rand(len(full_dS)) < 0.75
        Train = full_dS[msk]
        validate = full_dS[~msk]

        # Generating the model based on the feature list and target variable
        features = ['pickup_month','type_of_day', 'time_of_day','pickup_zip', 'dropoff_zip','pick_lat','pick_long','drop_lat','drop_long','distance_to_cover']
        x_train = Train[list(features)].values
        y_train = Train["trip_duration"].values
        x_validate = validate[list(features)].values
        y_validate = validate["trip_duration"].values
        x_test = test_ds[list(features)].values

        # this will generate a Decision tree regressor model on the provided data
        '''print("Decision tree regression modelling: ")
        regr_decision_tree = DecisionTreeRegressor(max_depth=10)
        regr_decision_tree.fit(x_train, y_train)
        with open('decision_tree_regression.pickle', 'wb') as handle:
            pickle.dump(regr_decision_tree, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('decision_tree_regression.pickle', 'rb') as handle:
            regr_decision_tree = pickle.load(handle)

        validation_result_decision_tree = regr_decision_tree.predict(x_validate)
        error_list = self.calculate_accuracy(y_validate, validation_result_decision_tree)
        print("Prediction Accuracy" + "   " + "No. of records")
        for i in range(5):
            print(str(i * 5) + "-" + str((i + 1) * 5) + " :" + str(error_list[i]))
            print()
        final_preiction_decision = regr_decision_tree.predict(x_test)
        print("Predicted trip duration by decision tree: "+str(final_preiction_decision[0]/60))
        print()'''


        # this will generate a random forest regressor model on the provided data
        print("Random forest  regression modelling: ")
        '''regr_random_forest = RandomForestRegressor(n_estimators=500, max_depth=10)
        print("Modelling starts")
        regr_random_forest.fit(x_train, y_train)
        with open('regression_model_lat_long_with_per_hr.pickle', 'wb') as handle:
            pickle.dump(regr_random_forest, handle, protocol=pickle.HIGHEST_PROTOCOL)'''

        with open('regression_model_lat_long_with_per_hr.pickle', 'rb') as handle:
            regr_random_forest = pickle.load(handle)

        validation_result = regr_random_forest.predict(x_validate)
        error_list = self.calculate_accuracy(y_validate, validation_result)
        final_status = regr_random_forest.predict(x_test)
        #print("Predicted trip duration by random forest: " + str(final_status[0] / 60))

        '''print("Prediction Accuracy " + "   " + "No. of records")
        for i in range(5):
            print(str(i * 5) + "-" + str((i + 1) * 5) + " :" + str(error_list[i]))
            print()
        feature_importance = regr_random_forest.feature_importances_
        print("Feature    " + " Importance")
        for i in range(len(feature_importance)):
            print(features[i] + " : " + str(feature_importance[i]))'''



        '''self.labelmain = Label(self.top, text=final_status[0], bg="black", fg="green", width=25, height=10)
        self.labelmain.pack()

        # Experimenting on changing number of trees for Random Trees
        mean_squared_error_chng_trees_l = []
        no_trees_l = []
        for i in range(1, 11):
            regr_random_forest = RandomForestRegressor(n_estimators=i * 5, max_depth=10)
            print("Modelling starts")
            regr_random_forest.fit(x_train, y_train)
            validation_result_tress = regr_random_forest.predict(x_validate)
            mean_squared_error_chng_trees_l.append(mean_squared_error(y_validate, validation_result_tress))

        print(mean_squared_error_chng_trees_l)

        # Experimenting on changing depth of trees keeping number of trees as 50 which has least MSE
        mean_squared_error_chng_depth_l = []
        for i in range(1,11):
            regr_random_forest = RandomForestRegressor(n_estimators=50, max_depth=i*2)
            print("Modelling starts")
            regr_random_forest.fit(x_train, y_train)
            validation_result_tress = regr_random_forest.predict(x_validate)
            mean_squared_error_chng_depth_l.append(mean_squared_error(y_validate, validation_result_tress))

        print(mean_squared_error_chng_depth_l)'''
        print( "The trip duration will be " + str(int(final_status[0] / 60)) + " minutes"   )
        return "The trip duration will be " + str(int(final_status[0] / 60)) + " minutes"

    def calculate_accuracy(self,actual,predicted):
        """
        This method calculates the accuracy of the model
        :param actual: Actual values of trip duration
        :param predicted: Predicted values of trip duration
        :return:
        """
        error_dict = {}
        error_list = []

        for i in range(0,11):
            error_dict[i] = 0

        for i in range(len(actual)):
            e = math.fabs(actual[i]-predicted[i])/actual[i]
            if 0 <= e <= 5:
                error_dict[0] += 1
            elif 5 < e <= 10:
                error_dict[1] += 1
            elif 10 < e <= 15:
                error_dict[2] += 1
            elif 15 < e <= 20:
                error_dict[3] += 1
            elif 20 < e <= 25:
                error_dict[4] += 1
            elif 25 < e <= 30:
                error_dict[5] += 1
            elif 30 < e <= 35:
                error_dict[6] += 1
            elif 35 < e <= 40:
                error_dict[7] += 1
            elif 40 < e <= 45:
                error_dict[8] += 1
            elif 45 < e <= 50:
                error_dict[9] += 1
            else:
                error_dict[10] += 1
        for values in error_dict.values():
            size = len(actual)
            error_list.append((values/size)*100)
        return error_list

# Used flask (a web application framework written in Python) for ceating the web server and talking with the web pages


@app.route('/TripDuration')
def start_page():
    """
    This is the start page
    :return: The rendered starting page
    """
    return render_template("TripDetails.html")


@app.route('/TripDuration', methods=['POST'])
def fetchingData():
    """
    This page takes data from the user and on submitting call the predicting_trip_duration() method
    in the trip_duration_prediction class
    :return: Rendered results
    """
    pick_address = request.form['pick']
    drop_address = request.form['drop']
    date = request.form['date']
    print()
    #print("Trip duration runnijng")
    tp = trip_duration_prediction()
    predicted_time = tp.predicting_trip_duration(pick_address, drop_address, date)
    #print(predicted_time)
    #print(pick_address)
    #print(drop_address)
    return render_template("TripDetails.html", trip_duration=predicted_time)
    # return redirect(url_for('success',name = user))'''

if __name__ == '__main__':
    """
    running the web server
    """
    app.debug = True
    app.run()
    app.run(debug=True)
