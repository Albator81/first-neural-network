import math


def clamp(x, a, b):
    if x < a: return a
    if x > b: return b
    return x


def get(x, y):
    return x if x is not None else y


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def add_all(l, func=None):
    if func is None:
        func = lambda e: e
    try:
        x = func(l[0])
    except:
        raise IndexError('Iterable is empty or doesnt support uint indexing')

    for i in range(1, len(l)):
        x = x + func(l[i])

    return x


class Connection:
    def __init__(self, src: 'Neuron', dst: 'Neuron', w=None) -> None:
        self.src = src
        self.dst = dst
        self.w = get(w, 1.)


class Neuron:
    act_func = sigmoid

    def __init__(self, v=None, b=None) -> None:
        self.val = get(v, 0.)
        self.bia = get(b, 0.)
        self.incoming = []  # connections to this neuron
        self.outgoing = []  # connections from this neuron

    def connect_to(self, receiver: 'Neuron', w=None):
        conn = Connection(self, receiver, w)
        self.outgoing.append(conn)
        receiver.incoming.append(conn)

    def t_activate(self, func=None):
        """Test activation, returns result. Doesn't change `self.val`"""
        f = get(func, Neuron.act_func)
        wv = lambda con: con.w * con.src.val
        x = add_all(self.incoming, wv)
        return f(x) + self.bia

    def activate(self, func=None):
        self.val = self.t_activate(func)


class InputNeuron (Neuron):
    def t_activate(self, func=None):
        return self.val + self.bia

    def activate(self, func=None):
        self.val = self.t_activate(func)


class OutputNeuron(Neuron): ...


class InnerNeuron (Neuron): ...
