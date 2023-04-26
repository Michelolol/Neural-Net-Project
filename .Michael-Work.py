from neural import NeuralNet
#what we want to use- estimate per 100 civilians, # registered, # unregistered, gun death rate, ,Suicide Rate by Firearm,Unintentional Deaths by Firearms, Rate Police Killing (per 10M)
#  ([Gp100Civ, GReg, GUnReg, GDeathRate, GSucideRate, GUninentDeath, GRatePolice], [RestrictVs!Restrict GLaws (1, 0)]),
training_data = [
    ([200, 2, 60, .75, 21, 32, 50000], [0]),
    ([200, 10, 40, 2000, 4000, 10000, 50000], [0]),
    ([2, .1, .01, .1, 1, 0, 10], [1]),
    ([1, .3, .01, .02, .1, 0, 5], [1]),
]



billy = NeuralNet(7, 15, 1)
billy.train(training_data)

#[Gp100Civ, GReg, GUnReg, GDeathRate, GSucideRate, GUninentDeath, GRatePolice],
test_data = [
    [200, 10, 400, 2000, 4000, 10000, 50000],
    [0, 0, 1, 0, 0, 0, 0],
]

print(f"case 1: {test_data[0]} evaluates to: {billy.evaluate(test_data[0])}")
print(f"case 2: {test_data[1]} evaluates to: {billy.evaluate(test_data[1])}")
#print(f"case 3: {test_data[2]} evaluates to: {billy.evaluate(test_data[2])}")
#print(f"case 4: {test_data[3]} evaluates to: {billy.evaluate(test_data[3])}")
#print(f"case 5: {test_data[4]} evaluates to: {billy.evaluate(test_data[4])}")







