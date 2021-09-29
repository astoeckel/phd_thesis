import nengo
import numpy as np

def lowpass_laplace(tau, order=0):
    scale = 1.0 / (tau ** order)
    numer = ((tau ** order) * scale,)
    denom = np.polynomial.Polynomial([tau, 1.0]) ** (order + 1)
    return (numer, tuple(denom.coef))

def LP(tau, order):
    return nengo.LinearFilter(*lowpass_laplace(tau, order), analog=True)

with nengo.Network() as model:
    #nd_in = nengo.Node(nengo.processes.WhiteSignal(period=100.0, high=1.0))
    nd_in = nengo.Node(lambda t: (t >= 1.0) * (t <= 2.0))
    ens0 = nengo.Ensemble(n_neurons=1, dimensions=1, neuron_type=nengo.Direct())
    ens1 = nengo.Ensemble(n_neurons=1, dimensions=1, neuron_type=nengo.Direct())

#    nengo.Connection(nd_in, ens0, synapse=LP(100e-3, 1), transform=-36.23)
#    nengo.Connection(nd_in, ens0, synapse=LP(120e-3, 1), transform=52.79)
#    nengo.Connection(ens0, ens0, synapse=LP(100e-3, 1), transform=-371.5544)
#    nengo.Connection(ens0, ens0, synapse=LP(120e-3, 1), transform=355.9951)

    # Integrator
    nengo.Connection(nd_in, ens0, synapse=LP(10e-3, 1), transform=0.0920)
    nengo.Connection(nd_in, ens0, synapse=LP(20e-3, 1), transform=-0.2572)
    nengo.Connection(nd_in, ens0, synapse=LP(30e-3, 1), transform=0.2788)
    nengo.Connection(ens0, ens0, synapse=LP(100e-3, 1), transform=16.8784)
    nengo.Connection(ens0, ens0, synapse=LP(110e-3, 1), transform=-27.5253)
    nengo.Connection(ens0, ens0, synapse=LP(120e-3, 1), transform=11.6470)

    # Pass-through with short time-constants
#    nengo.Connection(nd_in, ens0, synapse=LP(50e-3, 1), transform=26.7345)
#    nengo.Connection(nd_in, ens0, synapse=LP(51e-3, 1), transform=32.5549)
#    nengo.Connection(nd_in, ens0, synapse=LP(52e-3, 1), transform=41.5175)
#    nengo.Connection(ens0, ens0, synapse=LP(50e-3, 1), transform=-43.1146)
#    nengo.Connection(ens0, ens0, synapse=LP(51e-3, 1), transform=-34.2885)
#    nengo.Connection(ens0, ens0, synapse=LP(52e-3, 1), transform=-22.4031)

    # Pass-through with larger time-constants
#    nengo.Connection(nd_in, ens0, synapse=LP(100e-3, 1), transform=322.8709)
#    nengo.Connection(nd_in, ens0, synapse=LP(110e-3, 1), transform=-801.3733)
#    nengo.Connection(nd_in, ens0, synapse=LP(120e-3, 1), transform=512.1402)
#    nengo.Connection(ens0, ens0, synapse=LP(200e-3, 1), transform=7.7557)
#    nengo.Connection(ens0, ens0, synapse=LP(210e-3, 1), transform=-190.5015)
#    nengo.Connection(ens0, ens0, synapse=LP(220e-3, 1), transform=150.0776)


    nengo.Connection(nd_in, ens1, synapse=LP(100e-3, 1), transform=1.59)
    nengo.Connection(ens1, ens1, synapse=LP(200e-3, 1), transform=-1.1)

    nd_out = nengo.Node(size_in=3)
    nengo.Connection(ens0, nd_out[0], synapse=None)
    nengo.Connection(ens1, nd_out[1], synapse=None)
    #nengo.Connection(nd_in, nd_out[2], synapse=nengo.LinearFilter([1], [1, 0], analog=True))
    nengo.Connection(nd_in, nd_out[2], synapse=2e-3)