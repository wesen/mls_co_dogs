# from https://discuss.pytorch.org/t/print-autograd-graph/692/15

import torch
from graphviz import Digraph


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is None:
        params = {}
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(v):
        if v not in seen:
            if torch.is_tensor(v):
                dot.node(str(id(v)), size_to_str(v.size()), fillcolor='orange')
            elif hasattr(v, 'variable'):
                u = v.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(v)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(v)), str(type(v).__name__))
            seen.add(v)
            if hasattr(v, 'next_functions'):
                for u in v.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(v)))
                        add_nodes(u[0])
            if hasattr(v, 'saved_tensors'):
                for t in v.saved_tensors:
                    dot.edge(str(id(t)), str(id(v)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot
