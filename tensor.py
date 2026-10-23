from __future__ import annotations

from typing import Tuple, Self

import numpy as np

class Tensor:
    """
    Main class for Tensor
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: set = (),
        dtype: np.dtype = np.float32,
        copy: bool = False
    ):
        """
        Main tensor parameters and bacis data methods initialization

        Args:
            data (List[float]): Array for tensor falue
            requires_grad (bool): Whether gradient computation for this tensor is required or not
            _children (Tuple(Tensor)): Children tensors for topological graph building
            dtype (np.dtype): Type of tensor elements
            copy (bool): Flag for copying data if it is already numpy array
        
        Returns:
            out (Tensor): Initialized tensor with set parameters
        """

        if copy:
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = np.asarray(data, dtype=dtype)
    
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        self._backward = lambda: None
        self._previous = set(_children)

        self.__array_priority__ = 1000


    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the tensor
        """

        return self.data.ndim


    @property
    def shape(self) -> Tuple[int]:
        """
        Shape of the tensor
        """

        return self.data.shape


    @property
    def dtype(self) -> np.dtype:
        """
        Dtype of tensor values
        """
        
        return self.data.dtype


    @property
    def numel(self) -> int:
        """
        Length of the current view of tensor in elements
        """

        return int(np.prod(self.data))

    @property
    def size(
        self,
        dim: int | None = None
    ) -> Tuple[int] | int:
        """
        Size of a particular dimension or the shape of the current view

        Args:
            dim (int or None): Dimension which size to return
        """

        return self.data.size(dim)

    
    def item(self):
        """
        Takes scalar value from the tensor with one element

        Args:
            None
        
        Returns:
            Scalar: Scalar value from the tensor
        """

        if self.numel != 1:
            raise ValueError(f"a tensor with {self.numel} elements cannot be converted to scalar")

        return self.data[(0,) * self.ndim]


    def __hash__(self):
        """
        Computes identity of a tensor

        Args:
            None

        Returns:
            int: Hash value for tensor object
        """

        return id(self)
    

    def __len__(self):
        """
        Returns length of a tensor

        Args:
            None

        Returns:
            int: Length of a tensor
        """
        
        return len(self.data)
    

    def is_contiguous(self):
        """
        Identifies if the tensor has a contiguous area in memory

        Args:
            None

        Returns:
            bool: Contiguous flag
        """

        return self.data.flags.c_contiguous


    def backward(self):
        """
        Method for backward pass through all tensor operations and layers

        Args:
            None

        Returns:
            None
        """

        # building topological graph through all children
        graph = []
        visited = set()

        def build_graph(tensor):
            if tensor not in visited:
                visited.add(tensor)

                for child in tensor._previous:
                    build_graph(child)

                graph.append(tensor)

        build_graph(self)

        # backward pass of gradient with value of one through backward functions of all children
        self.grad = np.ones_like(self.data)

        for tensor in reversed(graph):
            tensor._backward()


    def __repr__(self) -> str:
        """
        Custom representation for better print

        Format: Data numpy view

        Args:
            None

        Returns:
            None
        """

        return f"{self.data}"
