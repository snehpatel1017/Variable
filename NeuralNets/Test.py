# In this File we have built a simple perceptrone using over Variable.py file in this we are predicting a simple line y = 2*x+2.
# You can predicte other mathemetical functions as well such as y= x**2 (Parabola)
# Note : depending on initianlization it may happen that you have to run the training multiple times for correct initialization.

from kiwisolver import Variable
from sklearn.linear_model import Perceptron


x_train = [[5.0],
           [4.0],
           [6.0],
           ]
y_train = [12.0, 10.0, 14.0]

p = Perceptron(1, ((1, 1),))
for itr in range(100):
    y_pred = []
    for x in x_train:
        y_pred.append(p(x))
    loss = Variable(0.0)
    for ypi, yi in zip(y_pred, y_train):
        loss += (ypi-yi)**2
    for ps in p.parameters():
        ps.grade = 0.0
    loss.backpropogation()

    print(loss.data)
    for ps in p.parameters():
        ps.data -= 0.01*ps.grade


cheking = p(x_train[0])
print(f"pridiction of first training set is : {cheking}")
