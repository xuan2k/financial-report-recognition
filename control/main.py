from networks import *
from utils.options import FOCROptions

class ReportRecognizer:
    def __init__(self, 
                 opts : FOCROptions) -> None:
        self.model = {}
        self.opts = opts
        
        for model_name in self.opts.model_list:

            self.model[model_name] = TableDetection(self.opts)

            

        