from typing import Any
from qadence import RydbergDevice, IdealDevice, Register
from networkx import Graph
from torch import Tensor


def get_default_register_topology(topology: str, **kwargs: Any) -> Graph:
    """
    Instantiates a register topology using an ideal Rydberg Device

    :param topology: chosen topology
    :param kwargs: arguments of necessary arguments to pass to subsequent loaders
    :return: a Register
    """
    # TODO: check if I need to allow for a device parameterization!
    rb_device: RydbergDevice = IdealDevice()

    if topology == 'triangular_lattice':
        reg = Register.triangular_lattice(device_specs=rb_device, **kwargs)
        return reg.graph
    elif topology == 'rectangular_lattice':
        reg = Register.rectangular_lattice(device_specs=rb_device, **kwargs)
        return reg.graph
    elif topology == 'line':
        reg = Register.line(device_specs=rb_device, **kwargs)
        return reg.graph
    elif topology == 'all_to_all':
        reg = Register.line(device_specs=rb_device, **kwargs)
        return reg.graph
    else:
        raise Exception(f"Unknown topology: {topology}")


def get_registers_from_sub_nodes(sub_nodes: Tensor, traps: Graph) -> Register:
    """
    Instantiates a Register object from a subset of nodes and their edges in the given traps graph.
    A register is the quantum system that will undergo evolution.

    :param sub_nodes: List of node indices to include in the subgraph.
    :param traps: The global graph representing possible traps.
    :return: A Register object representing the subgraph.
    """
    # Ensure all nodes in the list are present in the traps graph
    for node in sub_nodes:
        if node not in traps.nodes:
            raise ValueError(f"Node {node} is not in the traps graph")

    # Create the subgraph induced by the given nodes
    subgraph = traps.subgraph(sub_nodes)

    # Instantiate and return the Register object
    return Register(support=subgraph)
