import numpy as np
from functools import partial

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

def test_neuron(neuron, name='OP'):
    print('\n'+name)
    print('-----------------------------')
    print('x_1 \tx_2 \ty_hat')
    print('-----------------------------')
    for i in range(X.shape[0]):
      y_hat = neuron(X[i, :])
      print('{0} \t{1} \t{2}'.format(X[i, 0], X[i, 1], y_hat))
    
def escalon(z):
    if z >= 0.0:
        return 1.0
    else:
        return 0.0
    
def neuron(x, w, b):
  z = np.dot(w.T, x) + b
  a = escalon(z)

  return a

w_or = np.array([2, 2]).T
b_or = -1
or_neuron = partial(neuron, w=w_or, b=b_or)

w_nor = np.array([-1, -1]).T
b_nor = 0
nor_neuron = partial(neuron, w=w_nor, b=b_nor)

w_and = np.array([1, 1]).T
b_and = -2
and_neuron = partial(neuron, w=w_and, b=b_and)

def xnor_neuron(x):
    return or_neuron(np.array([and_neuron(x), nor_neuron(x)]))

test_neuron(and_neuron, name='AND')
test_neuron(or_neuron, name='OR')
test_neuron(nor_neuron, name='NOR')
test_neuron(xnor_neuron, name='XNOR')





  
