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
        copy: bool = False
    ):
        """
        Main tensor parameters and bacis data methods initialization

        Args:
            data (List[float]): Array for tensor falue
            requires_grad (bool): Whether gradient computation for this tensor is required or not
            _children (Tuple(Tensor)): Children tensors for topological graph building
        
        Returns:
            out (Tensor): Initialized tensor with set parameters
        """

        if copy:
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = np.asarray(data, dtype=np.float32)
    
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        # reinitializations of bacis methods using numpy array value
        self.shape = self.data.shape
        self.T = self.data.T
        self.ndim = self.data.ndim

        self._backward = lambda: None
        self._previous = set(_children)

        self.__array_priority__ = 1000

    
    def item(self):
        """
        Method for receiving value from tensor

        Args:
            None

        Returns:
            Numpy value from the tensor
        """

        return self.data


    def __hash__(self):
        """
        Compute identity of a tensor

        Args:
            None

        Returns:
            int: Hash value for tensor object
        """

        return id(self)
    

    def is_contiguous(self):
        """
        Identify if the tensor has a contiguous area in memory

        Args:
            None

        Returns:
            bool: Contiguous flag
        """

        return self.data.flags.c_contiguous
    

    def contiguous(self):
        """
        Make the tensor have a contiguous area in memory

        Args:
            None

        Returns:
            Tensor: contiguous tensor with initial data
        """

        if self.is_contiguous():
            return self
        
        out = Tensor(data=np.ascontiguousarray(self.data), requires_grad=self.requires_grad, _children=self._previous)

        out._backward = self._backward
        
        return out


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


    def __repr__(self):
        """
        Custom representation for better print

        Format: "Tensor(Data=data
                Grad=grad)"

        Args:
            None

        Returns:
            None
        """

        return f"Tensor(Data={self.data}\nGrad={self.grad})"
