## @ingroup Time_Accurate-Simulations
# save_prop_wake_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           

def save_prop_wake_vtk(VD,filename,Results,i_prop):
    # Extract wake properties of the ith propeller
    wVD             = VD.Wake
    n_time_steps    = len(wVD.XA1[i_prop,:,0,0])
    n_blades        = len(wVD.XA1[i_prop,0,:,0])
    n_radial_rings  = len(wVD.XA1[i_prop,0,0,:])
    n_props         = len(wVD.XA1[:,0,0,0])
    
    # Create file
    with open(filename, 'w') as f:
    
        #---------------------
        # Write header
        #---------------------
        l1 = "# vtk DataFile Version 4.0"               # File version and identifier
        l2 = "\nSUAVE Model of PROWIM Propeller Wake "  # Title 
        l3 = "\nASCII"                                  # Data type
        l4 = "\nDATASET UNSTRUCTURED_GRID"              # Dataset structure / topology   
        
        header = [l1, l2, l3, l4]
        f.writelines(header)   
        
        # --------------------
        # Write Points
        # --------------------   
        n_vertices = n_blades*(n_radial_rings+1)*(n_time_steps+1)    # total number of node vertices
        points_header = "\n\nPOINTS "+str(n_vertices) +" float"
        f.write(points_header)
        
        
        for B_idx in range(n_blades):
            for t_idx in range(n_time_steps+1):
                for r_idx in range(n_radial_rings+1):
                    # Get vertices for each node
                    if r_idx == n_radial_rings and t_idx==0:
                        # Last ring at t0; use B2 to get rightmost TE node
                        xp = round(wVD.XB2[i_prop,t_idx,B_idx,r_idx-1],4)
                        yp = round(wVD.YB2[i_prop,t_idx,B_idx,r_idx-1],4)
                        zp = round(wVD.ZB2[i_prop,t_idx,B_idx,r_idx-1],4)
                    elif t_idx==0:
                        # First set of rings; use A2 to get left TE node
                        xp = round(wVD.XA2[i_prop,t_idx,B_idx,r_idx],4)
                        yp = round(wVD.YA2[i_prop,t_idx,B_idx,r_idx],4)
                        zp = round(wVD.ZA2[i_prop,t_idx,B_idx,r_idx],4)                         
                    elif r_idx==n_radial_rings:  
                        # Last radial ring for tstep; use B1 of prior to get tip node
                        xp = round(wVD.XB1[i_prop,t_idx-1,B_idx,r_idx-1],4)
                        yp = round(wVD.YB1[i_prop,t_idx-1,B_idx,r_idx-1],4)
                        zp = round(wVD.ZB1[i_prop,t_idx-1,B_idx,r_idx-1],4)
                    else:
                        # print the point index (Left LE --> Left TE --> Right LE --> Right TE)
                        xp = round(wVD.XA1[i_prop,t_idx-1,B_idx,r_idx],4)
                        yp = round(wVD.YA1[i_prop,t_idx-1,B_idx,r_idx],4)
                        zp = round(wVD.ZA1[i_prop,t_idx-1,B_idx,r_idx],4)
                    
                    new_point = "\n"+str(xp)+" "+str(yp)+" "+str(zp)
                    node_number = r_idx + (n_radial_rings)*t_idx
                    #print("Point: ", new_point, "; Node Number: ", str(node_number))
                    f.write(new_point)                
        #---------------------    
        # Write Cells:
        #---------------------
        cells_per_blade = n_radial_rings*n_time_steps
        n_cells         = n_blades*cells_per_blade # total number of cells
        v_per_cell      = 4 # quad cells
        size            = n_cells*(1+v_per_cell) # total number of integer values required to represent the list
        cell_header     = "\n\nCELLS "+str(n_cells)+" "+str(size)
        f.write(cell_header)
        
        for B_idx in range(n_blades):
            for i in range(cells_per_blade):
                if i==0:
                    node = i + int(B_idx*n_vertices/n_blades)
                elif i%n_radial_rings ==0:
                    node = node+1
                new_cell = "\n4 "+str(node)+" "+str(node+1)+" "+str(node+n_radial_rings+2)+" "+str(node+n_radial_rings+1)
                f.write(new_cell)
                #print(new_cell)
                # update node:
                node = node+1 
        
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n_cells)
        f.write(cell_type_header)        
        for i in range(n_cells):
            f.write("\n9")       
            
        #--------------------------        
        # Write Scalar Cell Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n_cells)
        f.write(cell_data_header)      
        
        # First scalar value
        f.write("\nSCALARS circulation float 1")
        f.write("\nLOOKUP_TABLE default")   
        blade_circulation = Results['prop_outputs'].disc_circulation[0][0]
        for B_idx in range(n_blades):
            for i in range(cells_per_blade):
                circ_L = blade_circulation[int(i%(n_radial_rings))]
                circ_R = blade_circulation[int(i%(n_radial_rings))+1]
                circ_C = 0.5*(circ_L+circ_R)
                
                new_circ = str(circ_C)
                f.write("\n"+new_circ)     
        
        # Second scalar value
        f.write("\nSCALARS vt float 1")
        f.write("\nLOOKUP_TABLE default")   
        vt = Results['prop_outputs'].blade_tangential_induced_velocity[0]
        for B_idx in range(n_blades):
            for i in range(cells_per_blade):
                vt_L = vt[int(i%(n_radial_rings))]
                vt_R = vt[int(i%(n_radial_rings))+1]
                vt_C = 0.5*(vt_L+vt_R)
                
                new_vt = str(vt_C)
                f.write("\n"+new_vt)     
                
        ## Third scalar value
        #f.write("\nSCALARS Gamma float 1")
        #f.write("\nLOOKUP_TABLE default")  
        #g_end = int(len(Results['VD'].Wake_single_vector.GAMMA[0])/n_props)
        #Gamma_prop = Results['VD'].Wake_single_vector.GAMMA[0][0:g_end]
        #Gamma_blade = Gamma_prop[0:len(Gamma_prop)//n_blades]
        #for B_idx in range(n_blades):
            #for i in range(cells_per_blade):
                #gamma = Gamma_blade[i]
                
                #new_gamma = str(gamma)
                #f.write("\n"+new_gamma)            
                
    f.close()
        
        
    return