x, t = test[np.random.randint(len(test))]

# xnp = np.array(x[None])
xcupy = cupy.array(x[None])
predict = model.predictor(xcupy).array

# predict = model.predictor(x[None]).array
predict = predict[0][0]

if predict >= 0:
    print('Predicted Poisonous, Actual ' + ['Edible', 'Poisonous'][t[0]])
else:
    print('Predicted Edible, Actual ' + ['Edible', 'Poisonous'][t[0]])



# x, t = test[np.random.randint(len(test))]

# predict = model.predictor(x[None]).array
# predict = predict[0][0]

# if predict >= 0:
#     print('Predicted Poisonous, Actual ' + ['Edible', 'Poisonous'][t[0]])
# else:
#     print('Predicted Edible, Actual ' + ['Edible', 'Poisonous'][t[0]])
