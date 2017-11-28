import pandas as p # Importing Pandas library for using the data frame
import csv
from geopy.geocoders import Nominatim
import googlemaps
import time
from math import sin, cos, sqrt, atan2, radians
from random import randint
from datetime import datetime
from datetime import date
from datetime import time
from uszipcode import ZipcodeSearchEngine
import pickle # Importing Pickle library to store dictionary object


def distance_to_cover(pickup_point, drop_off_point):
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

def trip_prediction(df):
    geo_locator = Nominatim()
    now = datetime.now()
    api_key = 'AIzaSyCkyCJch-eC5EZBdgUx5lQwqknjwJ8hHOA'  # Google api_key
    gmaps = googlemaps.Client(api_key)

    # column_header = list(df)
    data_list = df.values.tolist()
    max_mile = -999999
    count = 0


    '''for row in data_list:
        del row[0]
        del row[1]
        row[0] = row[0].split(" ")
        dt = [int(x) for x in row[0][0].split("/")]
        day, month, year = dt[1], dt[0], 2016

        #date_ordinal = date(year, month, day).toordinal()
        day = date(year, month, day).isoweekday()

        if day == 6 or day == 7:
            row.insert(1, 1)  # weekend
        else:
            row.insert(1, 0)  # weekday

        t = [int(x) for x in row[0][1].split(":")]
        pickup_time = time(t[0], t[1], 0, 0, None)
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
        row.insert(2, time_of_day)

        #address_pick = str(row[3][0])+","+str(row[3][1])
        #location_address_pick = geo_locator.geocode(address_pick)
        #coordinates_pick = (location_address_pick.latitude, location_address_pick.longitude)
        #address_drop = str(row[4][0])+","+str(row[4][1])
        #location_address_drop = geo_locator.geocode(address_drop)
        #coordinates_drop = (location_address_drop.latitude, location_address_drop.longitude)

        search = ZipcodeSearchEngine()
        zipcode = search.by_coordinate(float(row[4]),float(row[3]), returns=1)  # Pickup zip
        if len(zipcode)> 0:
            pick_zipcode = int(zipcode[0]['Zipcode'])
            row.insert(3,pick_zipcode)  # pickup zip
        else:
            row.insert(3, -9999)

        zipcode = search.by_coordinate(float(row[7]), float(row[6]), returns=1)  # drop off zip
        if len(zipcode) > 0:
            drop_off_zipcode = int(zipcode[0]['Zipcode'])
            row.insert(4, drop_off_zipcode)  # drop_off_zipcode
        else:
            row.insert(4, -9999)  # drop_off_zipcode



        radius_earth = 6373.0
        latitude_pick = radians(row[6])
        longitude_pick = radians(row[5])
        latitude_drop = radians(row[8])
        longitude_drop = radians(row[7])
        differ_lon = longitude_drop - longitude_pick
        differ_lat = latitude_drop - latitude_pick
        a = (sin(differ_lat / 2)) ** 2 + cos(latitude_pick) * cos(latitude_drop) * (sin(differ_lon / 2)) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = radius_earth * c * 0.62137  # in miles

        row.insert(5,distance)
        row.insert(1,month)
        del row[0]
        print(count)
        count += 1
    with open('data_list_1.pickle', 'wb') as handle:
        pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    nyc_all_zipcodes = []
    with open("nyc_zipcodes.txt") as f:
        for row in f:
            l = row.split(",")
            for zip in l:
                if zip != '\n':
                    nyc_all_zipcodes.append(int(zip))

    with open('nyc_zip_codes.pickle', 'wb') as handle:
        pickle.dump(nyc_all_zipcodes, handle, protocol=pickle.HIGHEST_PROTOCOL)'''

    with open('nyc_zip_codes.pickle', 'rb') as handle:
        nyc_all_zipcodes = pickle.load(handle)

    with open('data_list_1.pickle', 'rb') as handle:
        data_list = pickle.load(handle)

    '''main_list = df.values.tolist()
    for i in range(len(main_list)):
        data_list[i].append(main_list[i][4])
        data_list[i].append(main_list[i][3])
        data_list[i].append(main_list[i][6])
        data_list[i].append(main_list[i][5])'''


    search = ZipcodeSearchEngine()



    column_header = ['pickup_month','type_of_day', 'time_of_day','pickup_zip', 'dropoff_zip','pick_lat','pick_long','drop_lat','drop_long','distance_to_cover','trip_duration']
    print("hello")
    with open("train_cleaned_new_full.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_header)
        writer.writeheader()
        size = len(data_list)
        for row_ind in range(size):
            if 100 <= data_list[row_ind][10] <= 10800 and data_list[row_ind][5] > .1:
                if data_list[row_ind][3] != -9999 and data_list[row_ind][4] != -9999 and data_list[row_ind][3] in nyc_all_zipcodes and data_list[row_ind][4] in nyc_all_zipcodes:
                    writer.writerow({column_header[0]: int(data_list[row_ind][0]), column_header[1]: int(data_list[row_ind][1]), column_header[2]: int(data_list[row_ind][2]),
                                 column_header[3]: int(data_list[row_ind][3]), column_header[4]: int(data_list[row_ind][4]), column_header[5]: float(data_list[row_ind][7]),
                                 column_header[6]: float(data_list[row_ind][6]),column_header[7]: float(data_list[row_ind][9]),column_header[8]: float(data_list[row_ind][8]),column_header[9]: float(data_list[row_ind][5]),column_header[10]: int(data_list[row_ind][10])})

    print(data_list[1])

def changing_data():
    file_name1 = 'train.csv'
    df1 = p.read_csv(file_name1)
    data_list1 = df1.values.tolist()

    file_name2 = 'train_cleaned_new.csv'
    df2 = p.read_csv(file_name2)
    data_list2 = df2.values.tolist()

    lat_long_list = []
    for row in data_list1:
        l = []
        l.append(row[5])
        l.append(row[4])
        l.append(row[7])
        l.append(row[6])
        l.append(row[8])
        lat_long_list.append(l)
    print(len(lat_long_list))

    print(len(data_list2))



def main():
    file_name = "train.csv"
    df = p.read_csv(file_name)
    geo_locator = Nominatim()
    now = datetime.now()
    api_key = 'AIzaSyCkyCJch-eC5EZBdgUx5lQwqknjwJ8hHOA'
    '''coordinates = input("enter the coordinates")
    location_coordinates = geo_locator.reverse(coordinates)
    print(location_coordinates.address)'''

    address_pick = input("Enter the pick up address: ")
    location_address_pick = geo_locator.geocode(address_pick)
    print("Address: " + location_address_pick.address)
    print("coordinates: " + str(location_address_pick.latitude) + "," + str(location_address_pick.longitude))
    coordinates_pick = (location_address_pick.latitude,location_address_pick.longitude)

    address_drop = input("Enter the drop off address: ")
    location_address_drop = geo_locator.geocode(address_drop)
    print("Address: " + location_address_drop.address)
    print("coordinates: " + str(location_address_drop.latitude) + "," + str(location_address_drop.longitude))
    coordinates_drop = (location_address_drop.latitude, location_address_drop.longitude)

    '''coordinates = input("enter the coordinates")
    location_coordinates = geo_locator.reverse(coordinates)
    print(location_coordinates.address)

    print(type(now))'''
    gmaps = googlemaps.Client(api_key)

    directions_result = gmaps.directions(coordinates_pick,
                                         coordinates_drop,
                                         mode="driving",
                                         avoid="ferries",
                                         departure_time=now
                                         )
    print(directions_result[0]['legs'][0]['distance']['text'])
    print(directions_result[0]['legs'][0]['duration']['text'])


#40.7639389,-73.97902679
#40.71008682,-74.00533295'''


    #trip_prediction(df)
    #changing_data()

if __name__ == '__main__':
    main()

