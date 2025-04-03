

class Complex():
    def __init__(self, *cochain_list, dimension):

        assert len(cochain_list) > 0
        if len(cochain_list) == 0:
            assert dimension == 0
        self.dimension = dimension
        self.cochains = {index : cochain_list[index] for index in range(len(cochain_list))}


    def consolidate(self, ):
        for dimension in range(self.dimension):
            if dimension > 0:
                lower_cells = self.cochains[dimension - 1]
                self.cochains[dimension].num_lower_cells = upper_cochain.num_cells
            if dimension < len(self.dimension):
                upper_cochain = self.cochains[dimension + 1]
                self.cochains[dimension].num_upper_cells = upper_cochain.num_cells
    


class BatchComplex():
    def __init__(self, *complexes):
        self.complexes = complexes
    
    def from_complexes():
        return

    