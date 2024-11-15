import pandas as pd
import numpy as np

class Undefined:
    #constructor method for ensuring that, when instanced,
    #a new object of this class has its own inizialized data
    def __init__(self,data):
        self.data = data
