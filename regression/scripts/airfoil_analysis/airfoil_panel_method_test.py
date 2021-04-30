# airfoil_panel_method_test.py
# 
# Created:  
# Modified: Mar 2021, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------

import SUAVE 
import os
from SUAVE.Core import Units, Data 
from SUAVE.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis import airfoil_analysis
import matplotlib.pyplot as plt   
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series \
     import  compute_naca_4series
from SUAVE.Plots.Airfoil_Plots  import plot_airfoil_properties

import numpy as np

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():    
    # Define Panelization 
    npanel = 160
    
    # Define Reynolds Number
    Re     = np.atleast_2d(np.array([5E5,5E6])).T
    
    #Define Angle of Attack
    AoA    = np.atleast_2d(np.linspace(-4,16,11)*Units.degrees).T 
     
    # -----------------------------------------------
    # SUAVE
    # -----------------------------------------------
    # Generate Airfoil Geometry (NACA 3310) 
    airfoil_geometry   = compute_naca_4series(0.03,0.3,0.1,npoints=npanel )
    
    # Compute Airfoil Aerodynamic and Boundary Layer Properties 
    # Batch Analysis: 
    airfoil_properties = airfoil_analysis(airfoil_geometry,AoA,Re, npanel, batch_analyis = True )  
    
    # Single Condition Analysis:  
    Re     = np.ones_like(AoA)*5E6  
    airfoil_properties_2 = airfoil_analysis(airfoil_geometry,AoA,Re, npanel, batch_analyis = False )  
    
    # Plot Results 
    plot_airfoil_properties(airfoil_properties,line_style='k-',arrow_color = 'r',plot_pressure_vectors = True)  
    
    
    # -----------------------------------------------
    # XFOIL
    # -----------------------------------------------
    xfoil_boundary_layer_file_1 = 'NACA_3310_4deg.txt'
    xfoil_cp_file_1             = 'NACA_3310_4deg_cp.txt' 
    xfoil_data_1                = read_xfoil_verification_files(xfoil_boundary_layer_file_1,xfoil_cp_file_1)  
    xfoil_data_1.Cl             = 0.7865
    xfoil_data_1.Cd             = 0.00507
    xfoil_data_1.Cm             = -0.0674
     
    # -----------------------------------------------
    # Validation  
    # ----------------------------------------------- 
    print('\n\nNACA 3310 Validation at 4 deg') 
    diff_CL           = np.abs(airfoil_properties.Cl[4,1] - xfoil_data_1.Cl) 
    expected_Cl_error = 0.21281484447415983
    print('\nCL difference')
    print(diff_CL)
    assert np.abs((airfoil_properties.Cl[4,1]  - xfoil_data_1.Cl)/xfoil_data_1.Cl) - expected_Cl_error < 1e-6
    
    diff_CD           = np.abs(airfoil_properties.Cd[4,1] - xfoil_data_1.Cd) 
    expected_Cd_error = 0.1719841745384137
    print('\nCD difference')
    print(diff_CD)
    assert np.abs((airfoil_properties.Cd[4,1]  - xfoil_data_1.Cd)/xfoil_data_1.Cd) - expected_Cd_error < 1e-6
    
    
    diff_CM           = np.abs(airfoil_properties.Cm[4,1] - xfoil_data_1.Cm) 
    expected_Cm_error = 0.385454350005437
    print('\nCM difference')
    print(diff_CM)
    assert np.abs((airfoil_properties.Cm[4,1]  - xfoil_data_1.Cm)/xfoil_data_1.Cm) - expected_Cm_error < 1e-6 
   
    diff_CP = np.abs(airfoil_properties.Cp[50,4,1]  - xfoil_data_1.Cp[50]) 
    expected_Cp_error = 0.05773430375306162
    print('\nCM difference')
    print(diff_CP)
    assert np.abs((airfoil_properties.Cp[50,4,1]  - xfoil_data_1.Cp[50])/xfoil_data_1.Cp[50]) - expected_Cp_error <  1e-6
    
    return  
 
# ----------------------------------------------------------------------
#  Read Xfoil Verification Files
# ----------------------------------------------------------------------
def read_xfoil_verification_files(xfoil_boundary_layer_file,xfoil_cp_file):
    """Reads xfoil data file for regression  
    
    Assumptions: 
    None 
    
    Inputs:  
    xfoil_boundary_layer_file
    xfoil_cp file             
        
    Outputs:
    airfoil_properties.
       x        - x coordinates of airfoil 
       y        - y coordinates of airfoil 
       Cp       - pressure coefficient 
       Ue/Vinf  - boundary layer edge velocity 
       Dstar    - displacement thickness 
       Theta    - momentum thickness 
       Cf       - friction coefficient 
       H        - boundary layer shape factor 
       H*       - 
       P        - 
       m        - 
       K        - 
       tau      - shear stress
       Di       - 
       
    Source:
    None
    
    """
    x_vals       = np.zeros(160)
    s_vals       = np.zeros_like(x_vals)
    y_vals       = np.zeros_like(x_vals)
    Cp_vals      = np.zeros_like(x_vals)
    Ue_Vinf_vals = np.zeros_like(x_vals)
    Dstar_vals   = np.zeros_like(x_vals)
    Theta_vals   = np.zeros_like(x_vals)
    Cf_vals      = np.zeros_like(x_vals)
    H_vals       = np.zeros_like(x_vals)
    H_star_vals  = np.zeros_like(x_vals) 
    P_vals       = np.zeros_like(x_vals) 
    
   
    with open(xfoil_boundary_layer_file,'r') as xblf:
        bl_file_lines  = xblf.readlines() 
        header         = 1
        for i in range(160):
            bl_line          = list(bl_file_lines[header + i][0:120].strip().split())   
            s_vals[i]        = float(bl_line[0])
            x_vals[i]        = float(bl_line[1])
            y_vals[i]        = float(bl_line[2]) 
            Ue_Vinf_vals[i]  = float(bl_line[3])
            Dstar_vals[i]    = float(bl_line[4])
            Theta_vals[i]    = float(bl_line[5])
            Cf_vals[i]       = float(bl_line[6])
            H_vals[i]        = float(bl_line[7])
            H_star_vals[i]   = float(bl_line[8])
            P_vals[i]        = float(bl_line[9]) 
            
    with open(xfoil_cp_file,'r') as xcpf:
        cp_file_lines  = xcpf.readlines() 
        header         = 1
        for i in range(160):
            cp_line       = list(cp_file_lines[header + i][0:120].strip().split())   
            Cp_vals[i]    = float(cp_line[1]) 
            
    airfoil_properties = Data(
        s          = s_vals,
        x          = x_vals,     
        y          = y_vals,     
        Cp         = Cp_vals,   
        Ue_Vinf    = Ue_Vinf_vals,
        delta_star = Dstar_vals,  
        theta      = Theta_vals, 
        Cf         = Cf_vals,     
        H          = H_vals,    
        H_star     = H_star_vals, 
        P          = P_vals,     
    )   
    
    return airfoil_properties 


if __name__ == '__main__': 
    main() 
    plt.show()