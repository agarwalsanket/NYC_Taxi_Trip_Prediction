#Importing the library and packages used
import pandas as pd
import numpy as np
import math
from math import sin, cos, sqrt, atan2, radians
from datetime import datetime
from datetime import date
from datetime import time
from uszipcode import ZipcodeSearchEngine
from geopy.geocoders import Nominatim
import pickle # Importing Pickle library to store dictionary object
from flask import Flask, redirect, url_for, request,render_template
app = Flask(__name__)


class trip_duration_prediction():


    def distance_to_cover(self,pickup_point, drop_off_point):
        radius_earth = 6373.0
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


        # the values entered are saved as a dataframe which will act as the testng dataframe for the prediction
        geo_locator = Nominatim()
        date_time_info = str(date_)
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
        time_of_day = None
        if pickup_time >= time(5, 0, 0, 0, None) and pickup_time <= time(8, 59, 0, 0, None):
            time_of_day = 0  # '5_to_8:59am'
        elif pickup_time >= time(9, 0, 0, 0, None) and pickup_time <= time(10, 59, 0, 0, None):
            time_of_day = 1  # '9_to_10:59am'
        elif pickup_time >= time(11, 0, 0, 0, None) and pickup_time <= time(15, 59, 0, 0, None):
            time_of_day = 2  # '11am_to_3:59pm'
        elif pickup_time >= time(16, 0, 0, 0, None) and pickup_time <= time(19, 59, 0, 0, None):
            time_of_day = 3  # '4pm_to_7:59pm'
        elif pickup_time >= time(20, 0, 0, 0, None) and pickup_time <= time(23, 59, 0, 0, None):
            time_of_day = 4  # '8pm_to_11:59pm'
        else:
            time_of_day = 5  # '00:00_to_4:59am'

        pickup_point_address = pickup
        print(pickup_point_address)

        location_address_pick = geo_locator.geocode(pickup_point_address)
        pick_lat = None
        pick_long = None
        drop_lat = None
        drop_long = None
        if location_address_pick is not None:
            pick_lat = location_address_pick.latitude
            pick_long = location_address_pick.longitude
            coordinates_pick = (pick_lat, pick_long)
        else:

            return "Enter proper pick up address"

        drop_off_address = drop
        location_address_drop_off = geo_locator.geocode(drop_off_address)
        if location_address_pick is not None:
            drop_lat = location_address_drop_off.latitude
            drop_long = location_address_drop_off.longitude
            coordinates_drop = (drop_lat,drop_long)
        else:
            return "Enter proper drop off address"

        search = ZipcodeSearchEngine()
        pickup_zip = search.by_coordinate(pick_lat, pick_long, returns=1)[0]['Zipcode']
        drop_off_zip = search.by_coordinate(drop_lat, drop_long, returns=1)[0]['Zipcode']

        distance = self.distance_to_cover(coordinates_pick, coordinates_drop)


        '''month = 11
        type_of_day = 1
        time_of_day = 2
        pick_lat = 40.6413
        pick_long = -73.7781
        drop_lat = 40.7831
        drop_long = -73.9712
        pickup_zip = search.by_coordinate(40.6413,-73.7781, returns=1)[0]['Zipcode']
        drop_off_zip = search.by_coordinate(40.7831,-73.9712, returns=1)[0]['Zipcode']
        pick_coord = (40.6413,-73.7781)
        drop_coord = (40.7831,-73.9712)
        distance = self.distance_to_cover(pick_coord, drop_coord)'''
        table = [month,type_of_day,time_of_day, pickup_zip,drop_off_zip,pick_lat,pick_long,drop_lat,drop_long,distance]

        testTable = [[int(month),int(type_of_day),int(time_of_day), int(pickup_zip),int(drop_off_zip),float(pick_lat),float(pick_long),float(drop_lat),float(drop_long), float(distance), '']]

        # cols will store names of attributes in the dataset
        cols = ['pickup_month','type_of_day', 'time_of_day','pickup_zip', 'dropoff_zip','pick_lat','pick_long','drop_lat','drop_long','distance_to_cover','trip_duration']
        df = pd.DataFrame(testTable, columns=cols)

        # FullDS will store the dataset provided on the given path
        full_dS = pd.read_csv('train_cleaned_new_full.csv')
        train = full_dS

        # tesing file will be the dataframe of details given by the user
        test_ds = df

        # Feature will have the list of features for modelling
        # Fragmenting the data into two parts: training set and validation set
        msk = np.random.rand(len(full_dS)) < 0.75
        Train = full_dS[msk]
        validate = full_dS[~msk]

        # Genrating the modle based on the feature list and target variable
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
        with open('regression_model_lat_long.pickle', 'wb') as handle:
            pickle.dump(regr_random_forest, handle, protocol=pickle.HIGHEST_PROTOCOL)'''

        with open('regression_model_lat_long.pickle', 'rb') as handle:
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
        return "The trip duration will be " + str(int(final_status[0] / 60)) + " minutes"



    def calculate_accuracy(self,actual,predicted):
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

@app.route('/predict')
def start_page():
    return render_template("TripDetails.html")

@app.route('/TripDuration', methods=['POST'])
def fetchingData():
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
    # return redirect(url_for('success',name = user))

if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug=True)




# 40.6413,73.7781
# 40.7831,73.9712