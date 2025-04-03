import torch
import torch_geometric 
import logging
from torch import Tensor
from torch_geometric.typing  import Adj
from torch_sparse import SparseTensor

class Cochain():
    def __init__(self, dimension : int = None, x: Tensor = None, upper_index : Adj = None, lower_index : Adj = None, 
                        boundary_index : Adj = None, shared_boundary_index : Tensor = None, 
                        shared_co_boundary_index : Tensor = None, upper_orient : Tensor = None, 
                        lower_orient : Tensor = None, **kwargs):

        
        self.__dimension__ = dimension
        self._x = x
        self.boundary_index = boundary_index
        self.upper_index = upper_index
        self.lower_index = lower_index
        self.shared_boundary_index = shared_boundary_index
        self.shared_co_boundary_index = shared_co_boundary_index
        self.upper_orient = upper_orient
        self.lower_orient = lower_orient

        for key, value in kwargs:
            if key == 'num_cells':
                self.__num_cells__ = value 
            elif key == "num_cells_down":
                self.num_cells_down = value 
            elif key == "num_cells_up":
                self.num_cells_up = value
            else:
                self[key] = value 

    @property
    def x(self):
        return self._x

    @property
    def dimension(self):
        return self.__dimension__

    @property
    def keys(self):
        return [key for key in self.__dict__.keys() if key[:2] != "__." and key[-2:] != '.__']
    
    @property
    def num_cells(self):
        return self.__num_cells__
    
    @property
    def down_cells(self):
        return self.num_down_cells
    
    @property
    def up_cells(self):
        return self.num_up_cells

    @property 
    def boundary_cells(self):
        return self.boundary_index
    
    @x.setter
    def set_x(self, new_x):
        if new_x:
            logging.warning("Deleting features")
        self._x = new_x

    @num_cells.setter
    def set_num_cells(self, n_cells):
        self.__num_cells__ = n_cells
    
    @dimension.setter
    def set_dimension(self, new_dimension):
        self.__dimension__ = new_dimension
    
    @down_cells.setter
    def set_num_cells(self, n_cells):
        self.__num_cells__ = n_cells

    @boundary_cells.setter
    def set_boundary_index(self, new_boundaary_index):
        self.set_boundary_index = new_boundaary_index
    
    def __getitem__(self, key):
        return self.__dict__.get(key, None)
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value 

    def __contains__(self, key):
        return key in self.__dict__.keys()


    @staticmethod
    def concatenation_dimension(key, item):
        if key in ["upper_index", "lower_index", "boundary_index", "shared_boundary", "shared_co_boundary_index"]:
            return -1 
        if isinstance(item, SparseTensor):
            return [0, 1]
        return 0
    
    @staticmethod
    def incrementing_value(self, key):
        if key in ["upper_index", "lower_index"]:
            if "__num_cells__" in self:
                return self.num_cells
            return self._x.size()[0]
        elif key == "shared_boundary_index":
            return self.num_cells_down
        elif key == "shared_co_boundary_index":
            return self.num_cells_up
        # handle boundary_index
    
        return 0
    

    def apply(self, function, item):
        if isinstance(item, list):
            return [function(item_) for item_ in item]
        if isinstance(item, dict):
            return {key : function(item_) for key, item_ in item.items()}
        if isinstance(item, SparseTensor):
            try:
                return function(item)
            except:
                return item
        return function(item)
    
    def apply_to_keys(self, function, *keys):
        for key in keys:
            self[key] = self.apply(function, self[key])
        return self
    
    def to(self, *keys, device):
            self.apply_to_keys(lambda x : x.to(device), *keys)

    def contigous(self, *keys):
        self.apply_to_keys(lambda x : x._countigous(), *keys)
    

class CochainBatch(Cochain):

    ''''
    constructs a batch from cochain list
    '''

    def __init__(self, dimension, **kwargs):
        super(CochainBatch,).__init__()
        self.dimension = dimension # dimension of the cochains

    def from_cochain_list(self, cls, *cochain_list : Cochain): # Constructs CochainBatch from a list of cochains

        keys = set([key for data in cochain_list for key in data.keys()]) # get all the keys from the cochains
        batch = cls(cochain_list[0].__dimension__) #
        batch.ptr = [0]
        self.num_cochains = len(cochain_list)
        batch.__slices__ = {key : [0] for key in keys}
        batch.__cumsum__ = {key : [0] for key in keys}
        batch.__cat_dims__ = {}

        for cochain in cochain_list:
            for key in cochain.keys():
                if cochain[key] is not None:

                    value = cochain[key]

                    current_cum = batch.__cumsum__[key][-1] # last current_cum

                    if isinstance(current_cum, Tensor) and current_cum != 0 and current_cum.dtype != bool:
                        value += current_cum
                    elif isinstance(current_cum, int) and current_cum != 0 and current_cum.dtype != bool:
                        value += current_cum
                    elif isinstance(current_cum, SparseTensor):
                        current_value = current_cum.storage.value()
                        if isinstance(current_value, int) and current_value != 0 and current_value.dtype != bool:
                            new_value = current_cum + current_value
                            current_cum.set_value(new_value)
                    
                    batch.__cumsum__[key].append(current_cum)



        





