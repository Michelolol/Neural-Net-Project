from neural import NeuralNet 
import csv
import pandas
import numpy as np

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

    for col in data_fix:
        if col.startswith('Unintentional Deaths'):
            print("I found", col, "and am now turning them into doubles")
            data_fix[col] = data_fix[col].astype(float)
            
    
    print(data_fix)
    
    
    
    #data = csv.reader(data_fix)



#what we want to use- estimate per 100 civilians, # registered, # unregistered, gun death rate, ,Suicide Rate by Firearm,Unintentional Deaths by Firearms, Rate Police Killing (per 10M)
#  ([Gp100Civ, GReg, GUnReg, GDeathRate, GSucideRate, GUninentDeath, GRatePolice], [RestrictVs!Restrict GLaws (1, 0)]),
#firearms civ possess,Reg firearms,Unreg firearms,Gun Death Rate,Suicide Rate, Unintentional Deaths, Police Killing (per 10M),The regulation of guns
training_data = [
    ([40000, 10000, 10000, 10000, 10000, 10000, 50000], [0]),
    ([33000, 10000, 10000, 210000000, 4000, 10000, 50000], [0]),
    ([2, .1, .01, .1, 1, 0, 10], [1]),
    ([1, .3, .01, .02, .1, 0, 5], [1]),
    ([0, .0, .00, .00, 0, 0, 0], [1]),
]



billy = NeuralNet(7, 50, 1)                                                                                                                                                                                                                                                                                                                                                                           
billy.train(training_data)

#[Gp100Civ, GReg, GUnReg, GDeathRate, GSucideRate, GUninentDeath, GRatePolice],
test_data = [
    [10000, 10000, 10000, 10000, 40000, 10000, 50000],
    [0, 0, 0, 0, 0, 0, 0],
    [33000.0,0.0,0.0,19.29,2.37,0.68,0.0], #america, should be 0

]


print(f"case 1: {test_data[0]} evaluates to: {billy.evaluate(test_data[0])}")
print(f"case 2: {test_data[1]} evaluates to: {billy.evaluate(test_data[1])}")
#print(f"case 3: {test_data[2]} evaluates to: {billy.evaluate(test_data[2])}")
#print(f"case 4: {test_data[3]} evaluates to: {billy.evaluate(test_data[3])}")
#print(f"case 5: {test_data[4]} evaluates to: {billy.evaluate(test_data[4])}")







