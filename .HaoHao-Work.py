from neural import NeuralNet
import csv
import pandas
import numpy as np
#what we want to use- estimate per civilians, # registered, # unregistered, gun death rate

with open('dataset_gun.csv', 'r') as csv_file:
    
    data_fix = pandas.read_csv('dataset_gun.csv')

    print(data_fix)
    data_fix.pop('Country')
    data_fix.pop('Population')
    data_fix.pop('firearms per 100 persons')
    data_fix.pop('Computation method')
    data_fix.pop('Police Killings')
    data_fix.pop('Deaths by firearm')
    data_fix.pop('Data Year Police Killing')
    data_fix.pop('Notes')
    data_fix.replace(to_replace='restrictive',value=1.0, inplace= True)
    data_fix.replace(to_replace='permissive',value=0.0, inplace= True)
    data_fix.replace(to_replace=np.NaN,value=0.0, inplace= True)
    data_fix.replace(to_replace=' per 100k',value='', inplace= True, regex= True)
    data_fix.replace(to_replace='–',value='0.0', inplace= True, regex= True)

    for col in data_fix:
        if col.startswith('firearms in civ'):
            print("I found", col, "and am now turning them into doubles")
            data_fix[col] = data_fix[col].astype(float)
        if col.startswith('Unintentional Deaths'):
            print("I found", col, "and am now turning them into doubles")
            data_fix[col] = data_fix[col].astype(float)
        if col.startswith('Suicide Rate by'):
            print("I found", col, "and am now turning them into doubles")
            data_fix[col] = data_fix[col].astype(float)
            
    
    print(data_fix)
    
    
    
    #data = csv.reader(data_fix)
column_to_move = data_fix.pop("The regulation of guns")

data_fix['The regulation of guns'] = column_to_move

print(data_fix)
for row in data_fix:
    print(data_fix[row])



data_fix.to_csv('FixedGunData.csv')

csv_filename = 'FixedGunData.csv'
with open(csv_filename) as f:
    reader = csv.reader(f)
    lst = list(tuple(line) for line in reader)

simple_data = [item for t in lst for item in t]


print(simple_data)

billy = NeuralNet(7, 50, 1)                                                                                                                                                                                                                                                                                                                                                                           
billy.train(lst)