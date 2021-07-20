## @ingroup Components-Energy-Converters
# Propeller.py
#
# Created:  July 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Rotor import Rotor

# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Converters
class Propeller(Rotor):
    """This is a propeller component.
    
    Assumptions:
    None

    Source:
    None
    """     
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """         
        
        self.tag      = 'propeller'

        


