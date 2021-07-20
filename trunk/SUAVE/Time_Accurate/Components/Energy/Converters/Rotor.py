## @ingroup Time_Accurate-Components-Energy-Converters
# Rotor.py
#
# Created:  Jul 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Geometry.Three_Dimensional \
     import  orientation_product, orientation_transpose

# package imports
import numpy as np
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------
#  Generalized Rotor Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Converters
class Rotor(Energy_Component):
    """This is a general rotor component for time-accurate simulations.
    
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

        self.tag                       = 'rotor'
        self.number_of_blades          = 0.0 
        self.tip_radius                = 0.0
        self.hub_radius                = 0.0
        self.twist_distribution        = 0.0
        self.chord_distribution        = 0.0
        self.thickness_to_chord        = 0.0
        self.mid_chord_alignment       = 0.0
        self.blade_solidity            = 0.0
        self.thrust_angle              = 0.0
        self.pitch_command             = 0.0 
        self.design_power              = None
        self.VTOL_flag                 = False
        self.design_thrust             = None
        self.induced_hover_velocity    = 0.0
        self.airfoil_geometry          = None
        self.airfoil_polars            = None
        self.airfoil_polar_stations    = None 
        self.radius_distribution       = None
        self.rotation                  = [1]      # counter-clockwise rotation as viewed from the front of the aircraft 
        self.ducted                    = False         
        self.number_azimuthal_stations = 24
        self.induced_power_factor      = 1.48     # accounts for interference effects
        self.profile_drag_coefficient  = .03 
        self.use_2d_analysis           = False    # True if rotor is at an angle relative to freestream or nonuniform freestream
        self.nonuniform_freestream     = False
        self.axial_velocities_2d       = None     # allows for additional velocity influences at the rotor
        self.tangential_velocities_2d  = None     # allows for additional velocity influences at the rotor
        self.radial_velocities_2d      = None     # allows for additional velocity influences at the rotor
        
        # Time-Accurate Properties
        self.azimuthal_offset          = 0.0      # [rad] Starting position of rotor, 0 indicates first blade points toward starboard wing
        self.timesteps_per_revolution  = 200      # Number of timesteps per revolution
        self.number_of_revolutions  = 1        # Number rotor revolutions

    def spin(self,conditions):
        """Analyzes a general rotor given geometry and operating conditions
        and provides instantaneous forces of the rotor.

        Assumptions:
        per source

        Source:
        Drela, M. "Qprop Formulation", MIT AeroAstro, June 2006
        http://web.mit.edu/drela/Public/web/qprop/qprop_theory.pdf
        
        Leishman, Gordon J. Principles of helicopter aerodynamics
        Cambridge university press, 2006.      

        Inputs:
        self.inputs.omega                    [radian/s]
        conditions.freestream.               
          density                            [kg/m^3]
          dynamic_viscosity                  [kg/(m-s)]
          speed_of_sound                     [m/s]
          temperature                        [K]
        conditions.frames.                   
          body.transform_to_inertial         (rotation matrix)
          inertial.velocity_vector           [m/s]
        conditions.propulsion.               
          throttle                           [-]
                                             
        Outputs:                             
        conditions.propulsion.outputs.        
           number_radial_stations            [-]
           number_azimuthal_stations         [-] 
           disc_radial_distribution          [m]
           thrust_angle                      [rad]
           speed_of_sound                    [m/s]
           density                           [kg/m-3]
           velocity                          [m/s]
           disc_tangential_induced_velocity  [m/s]
           disc_axial_induced_velocity       [m/s]
           disc_tangential_velocity          [m/s]
           disc_axial_velocity               [m/s]
           drag_coefficient                  [-]
           lift_coefficient                  [-]
           omega                             [rad/s]
           disc_circulation                  [-] 
           blade_dQ_dR                       [N/m]
           blade_dT_dr                       [N]
           blade_thrust_distribution         [N]
           disc_thrust_distribution          [N]
           thrust_per_blade                  [N]
           thrust_coefficient                [-] 
           azimuthal_distribution            [rad]
           disc_azimuthal_distribution       [rad]
           blade_dQ_dR                       [N]
           blade_dQ_dr                       [Nm]
           blade_torque_distribution         [Nm] 
           disc_torque_distribution          [Nm] 
           torque_per_blade                  [Nm] 
           torque_coefficient                [-] 
           power                             [W]    
           power_coefficient                 [-] 
                                             
        Properties Used:                     
        self.                                
          number_of_blades                   [-]
          tip_radius                         [m]
          twist_distribution                 [radians]
          chord_distribution                 [m]
          radius_distribution                [m]
          thickness_to_chord                 [-]
          inputs.
             omega                           [rad/sec]
          airfoil_geometry                   [string or None]
          airfoil_polar_stations             [string or None]
          airfoil_cl_surrogates              [-]
          airfoil_cd_surrogates              [-]
          induced_hover_velocity             [-]
          thrust_angle                       [radians]
          
        """         
        #----------------------------------------------------------------------------------
        # Unpack parameters 
        #----------------------------------------------------------------------------------   
        
        # Unpack rotor blade parameters
        B       = self.number_of_blades 
        R       = self.tip_radius
        beta_0  = self.twist_distribution
        c       = self.chord_distribution
        r_1d    = self.radius_distribution 
        tc      = self.thickness_to_chord
        
        # Unpack rotor airfoil data
        a_geo   = self.airfoil_geometry      
        a_loc   = self.airfoil_polar_stations  
        cl_sur  = self.airfoil_cl_surrogates
        cd_sur  = self.airfoil_cd_surrogates 
        
        # Unpack rotor inputs and conditions
        omega                 = self.inputs.omega
        ua_self               = self.induced_hover_velocity
        pitch_c               = self.pitch_command
        theta                 = self.thrust_angle 
        Na                    = self.number_azimuthal_stations 
        nonuniform_freestream = self.nonuniform_freestream
        use_2d_analysis       = self.use_2d_analysis
        rotation              = self.rotation
        start_angle           = self.azimuthal_offset
        steps_per_rev         = self.timesteps_per_revolution
        nrevs                 = self.number_of_revolutions
        Nsteps                = steps_per_rev*nrevs
        
        # Set blade offsets for each time step
        age               = np.linspace(0, 2*np.pi*nrevs, Nsteps)
        azimuthal_offset  = start_angle + age
        
        # Unpack freestream conditions, vectorize for number of wake time steps
        rho     = conditions.freestream.density
        mu      = conditions.freestream.dynamic_viscosity
        Vv      = conditions.frames.inertial.velocity_vector
        a       = conditions.freestream.speed_of_sound
        T       = conditions.freestream.temperature
        nu      = mu/rho 
        rho_0   = rho
        
        #----------------------------------------------------------------------------------
        # Update blade rotations and azimuthal offsets
        #----------------------------------------------------------------------------------
        # check for invalid rotation angle
        if (rotation == [1]) or (rotation == [-1]):  
            pass
        else:
            print("Invalid rotation direction. Setting to [1].")
            rotation = [1]
        # check for invalid rotation rate   
        if np.any(omega) <0:
            print("Negative rotation rate defined! Reverting direction.")
            omega    = np.abs(omega)
            rotation = -rotation
        
        # update blade position with prescribed offset value
        if rotation == [1]:
            # clockwise rotation about x-axis, negate the offset
            blade_offsets   = np.repeat(azimuthal_offset[:,None],B,axis=1) + np.linspace(0,2*np.pi,B+1)[0:-1]
        else:
            blade_offsets   = azimuthal_offset + np.linspace(0,2*np.pi,B+1)[0:-1]
        
        # blade offsets independent of rotation number, so get base rotation angle
        blade_offsets = blade_offsets%2*np.pi
              
        #----------------------------------------------------------------------------------
        # Orient velocity frame
        #----------------------------------------------------------------------------------  
        
        # Velocity in the body frame
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_inertial2body = orientation_transpose(T_body2inertial)
        V_body          = orientation_product(T_inertial2body,Vv)
        body2thrust     = np.array([[np.cos(theta), 0., np.sin(theta)],[0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        T_body2thrust   = orientation_transpose(np.ones_like(T_body2inertial[:])*body2thrust)  
        V_thrust        = orientation_product(T_body2thrust,V_body) 
        
        # Account for self-induced hover velocity if nonzero
        V  = V_thrust[:,0,None] + ua_self
        
        #----------------------------------------------------------------------------------
        # Initialize for BEMT
        #----------------------------------------------------------------------------------
        # Things that don't change with iteration
        Nr       = len(c)    # Number of radial stations along single propeller blade    
        ctrl_pts = len(Vv)   # Number of control points / conditions for evaluation of performance
        
        # Helpful shorthands
        BB      = B*B    
        BBB     = BB*B
        pi      = np.pi            
        pi2     = pi*pi         
        
        # calculate total blade pitch
        total_blade_pitch = beta_0 + pitch_c  
        
        # Non-dimensional radial distribution and differential radius 
        chi          = r_1d/R
        diff_r       = np.diff(r_1d)
        deltar       = np.zeros(len(r_1d))
        deltar[1:-1] = diff_r[0:-1]/2 + diff_r[1:]/2
        deltar[0]    = diff_r[0]/2
        deltar[-1]   = diff_r[-1]/2

        # Calculating rotational parameters
        omegar   = np.outer(omega,r_1d)
        n        = omega/(2.*pi)       # [Rot/sec]
        
        # Azimuthal distribution of stations
        psi            = np.linspace(0,2*pi,Na+1)[:-1]
        psi_2d         = np.tile(np.atleast_2d(psi).T,(1,Nr))
        psi_2d         = np.repeat(psi_2d[None,:,:], ctrl_pts, axis=0) 
        
        # 2D radial distribution, dimensional and non-dimensionalized
        chi_2d         = np.tile(chi ,(Na,1))            
        chi_2d         = np.repeat(chi_2d[None,:,:], ctrl_pts, axis=0) 
        r_dim_2d       = np.tile(r_1d ,(Na,1))  
        r_dim_2d       = np.repeat(r_dim_2d[None,:,:], ctrl_pts, axis=0)          
        c_2d           = np.tile(c ,(Na,1)) 
        c_2d           = np.repeat(c_2d[None,:,:], ctrl_pts, axis=0)  
     
        #----------------------------------------------------------------------------------        
        # Include non-uniform velocity disturbances of entire disc
        #----------------------------------------------------------------------------------
        
        # starting with uniform freestream assumption
        ua  = 0             
        ut  = 0            
        ur  = 0     
              
        # Include velocities introduced by rotor incidence angle
        if theta !=0:
            # incidence angle creates disturbances in radial and tangential velocities
            use_2d_analysis  = True
            
            # component of vertical freestream in the rotor plane
            Vz  = V_thrust[:,2,None,None]
            Vz  = np.repeat(Vz, Na,axis=1)
            Vz  = np.repeat(Vz, Nr,axis=2)
            
            # compute resulting radial and tangential velocities in rotor frame
            ut +=  ( Vz*np.cos(psi_2d)  ) * rotation[0]
            ur +=  (-Vz*np.sin(psi_2d)  )
            ua +=  np.zeros_like(ut) 
            
        # Include additional user-defined velocities (ie. headwinds, upstream influences, etc.)
        if nonuniform_freestream:
            use_2d_analysis   = True

            # include additional influences specified at rotor sections, shape=(ctrl_pts,Na,Nr)
            ua += self.axial_velocities_2d
            ut += self.tangential_velocities_2d
            ur += self.radial_velocities_2d

        #----------------------------------------------------------------------------------        
        # Make things 2D if there is any nonuniformity in the propeller inflow
        #----------------------------------------------------------------------------------
        if use_2d_analysis:
            # make everything 2D with shape (ctrl_pts,Na,Nr)
            size   = (ctrl_pts,Na,Nr)
            PSI    = np.ones(size)
            PSIold = np.zeros(size)
            
            # 2-D streamwise freestream velocity and omega*r
            V_2d   = V_thrust[:,0,None,None]
            V_2d   = np.repeat(V_2d, Na,axis=1)
            V_2d   = np.repeat(V_2d, Nr,axis=2)
            omegar = (np.repeat(np.outer(omega,r_1d)[:,None,:], Na, axis=1))
            
            # total velocities
            Ua     = V_2d + ua         
            
            # 2-D blade pitch and radial distributions
            beta = np.tile(total_blade_pitch,(Na ,1))
            beta = np.repeat(beta[np.newaxis,:, :], ctrl_pts, axis=0)
            r    = np.tile(r_1d,(Na ,1))
            r    = np.repeat(r[np.newaxis,:, :], ctrl_pts, axis=0) 
            
            # 2-D atmospheric properties
            a   = np.tile(np.atleast_2d(a),(1,Nr))
            a   = np.repeat(a[:, np.newaxis,  :], Na, axis=1)  
            nu  = np.tile(np.atleast_2d(nu),(1,Nr))
            nu  = np.repeat(nu[:, np.newaxis,  :], Na, axis=1)    
            rho = np.tile(np.atleast_2d(rho),(1,Nr))
            rho = np.repeat(rho[:, np.newaxis,  :], Na, axis=1)  
            T   = np.tile(np.atleast_2d(T),(1,Nr))
            T   = np.repeat(T[:, np.newaxis,  :], Na, axis=1)              
            
        else:
            # total velocities
            r      = r_1d
            Ua     = np.outer((V + ua),np.ones_like(r))
            beta   = total_blade_pitch
            
            # Things that will change with iteration
            size   = (ctrl_pts,Nr)
            PSI    = np.ones(size)
            PSIold = np.zeros(size)  
        
        # total velocities
        Ut   = omegar - ut
        U    = np.sqrt(Ua*Ua + Ut*Ut + ur*ur)
        
        # Setup a Newton iteration
        diff   = 1. 
        ii     = 0
        tol    = 1e-6  # Convergence tolerance        
        
        #----------------------------------------------------------------------------------        
        # BEMT analysis with Newton iteration to converge on circulation at the blade
        #----------------------------------------------------------------------------------
        while (diff>tol):
            sin_psi      = np.sin(PSI)
            cos_psi      = np.cos(PSI)
            Wa           = 0.5*Ua + 0.5*U*sin_psi
            Wt           = 0.5*Ut + 0.5*U*cos_psi   
            va           = Wa - Ua
            vt           = Ut - Wt
            alpha        = beta - np.arctan2(Wa,Wt)
            W            = (Wa*Wa + Wt*Wt)**0.5
            Ma           = (W)/a        # a is the speed of sound  
            lamdaw       = r*Wa/(R*Wt) 
            
            # Limiter to keep from Nan-ing
            lamdaw[lamdaw<0.] = 0. 
            f            = (B/2.)*(1.-r/R)/lamdaw
            f[f<0.]      = 0.
            piece        = np.exp(-f)
            arccos_piece = np.arccos(piece)
            F            = 2.*arccos_piece/pi # Prandtl's tip factor
            Gamma        = vt*(4.*pi*r/B)*F*(1.+(4.*lamdaw*R/(pi*B*r))*(4.*lamdaw*R/(pi*B*r)))**0.5 
            Re           = (W*c)/nu  
            
            # Compute aerodynamic forces based on specified input airfoil or using a surrogate
            Cl, Cdval = compute_aerodynamic_forces(a_loc, a_geo, cl_sur, cd_sur, ctrl_pts, Nr, Na, Re, Ma, alpha, tc, use_2d_analysis)

            Rsquiggly   = Gamma - 0.5*W*c*Cl
        
            # An analytical derivative for dR_dpsi, this is derived by taking a derivative of the above equations
            # This was solved symbolically in Matlab and exported        
            f_wt_2      = 4*Wt*Wt
            f_wa_2      = 4*Wa*Wa
            Ucospsi     = U*cos_psi
            Usinpsi     = U*sin_psi
            Utcospsi    = Ut*cos_psi
            Uasinpsi    = Ua*sin_psi 
            UapUsinpsi  = (Ua + Usinpsi)
            utpUcospsi  = (Ut + Ucospsi) 
            utpUcospsi2 = utpUcospsi*utpUcospsi
            UapUsinpsi2 = UapUsinpsi*UapUsinpsi 
            dR_dpsi     = ((4.*U*r*arccos_piece*sin_psi*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5))/B - 
                           (pi*U*(Ua*cos_psi - Ut*sin_psi)*(beta - np.arctan((Wa+Wa)/(Wt+Wt))))/(2.*(f_wt_2 + f_wa_2)**(0.5))
                           + (pi*U*(f_wt_2 +f_wa_2)**(0.5)*(U + Utcospsi  +  Uasinpsi))/(2.*(f_wa_2/(f_wt_2) + 1.)*utpUcospsi2)
                           - (4.*U*piece*((16.*UapUsinpsi2)/(BB*pi2*f_wt_2) + 1.)**(0.5)*(R - r)*(Ut/2. - 
                            (Ucospsi)/2.)*(U + Utcospsi + Uasinpsi ))/(f_wa_2*(1. - np.exp(-(B*(Wt+Wt)*(R - 
                            r))/(r*(Wa+Wa))))**(0.5)) + (128.*U*r*arccos_piece*(Wa+Wa)*(Ut/2. - (Ucospsi)/2.)*(U + 
                            Utcospsi  + Uasinpsi ))/(BBB*pi2*utpUcospsi*utpUcospsi2*((16.*f_wa_2)/(BB*pi2*f_wt_2) + 1.)**(0.5))) 
        
            dR_dpsi[np.isnan(dR_dpsi)] = 0.1
        
            dpsi        = -Rsquiggly/dR_dpsi
            PSI         = PSI + dpsi
            diff        = np.max(abs(PSIold-PSI))
            PSIold      = PSI
        
            # If omega = 0, do not run BEMT convergence loop 
            if all(omega[:,0]) == 0. :
                print("Propeller BEMT did not run convergence loop (0rpm)")
                break
            
            # If its really not going to converge
            if np.any(PSI>pi/2) and np.any(dpsi>0.0):
                print("Propeller BEMT did not converge to a solution (Stall)")
                break
        
            ii+=1 
            if ii>10000:
                print("Propeller BEMT did not converge to a solution (Iteration Limit)")
                break
    
        # More Cd Mach scaling from AA241ab notes for turbulent skin friction
        Tw_Tinf     = 1. + 1.78*(Ma*Ma)
        Tp_Tinf     = 1. + 0.035*(Ma*Ma) + 0.45*(Tw_Tinf-1.)
        Tp          = (Tp_Tinf)*T
        Rp_Rinf     = (Tp_Tinf**2.5)*(Tp+110.4)/(T+110.4) 
        Cd          = ((1/Tp_Tinf)*(1/Rp_Rinf)**0.2)*Cdval  
        
        epsilon                  = Cd/Cl
        epsilon[epsilon==np.inf] = 10. 

        blade_T_distribution     = rho*(Gamma*(Wt-epsilon*Wa))*deltar 
        blade_Q_distribution     = rho*(Gamma*(Wa+epsilon*Wt)*r)*deltar 
        
        if use_2d_analysis:
            # velocity and force distribution along rotor disc as a function of azimuth and radial locations
            blade_T_distribution_disc = blade_T_distribution
            blade_Q_distribution_disc = blade_Q_distribution      
            blade_Gamma_disc          = Gamma      
            Va_disc                   = Wa
            Vt_disc                   = Wt
            Va_ind_disc               = va
            Vt_ind_disc               = vt            
            
            # Close the loop to enable interpolation along last azimuthal slice
            psi_closed                       = np.append(psi,2*np.pi)
            blade_T_distribution_disc_closed = np.append(blade_T_distribution_disc,blade_T_distribution_disc[:,0,:][:,None,:],axis=1)
            blade_Q_distribution_disc_closed = np.append(blade_Q_distribution_disc,blade_Q_distribution_disc[:,0,:][:,None,:],axis=1)
            blade_Gamma_disc_closed          = np.append(blade_Gamma_disc,blade_Gamma_disc[:,0,:][:,None,:],axis=1)
            Va_disc_closed                   = np.append(Va_disc,Va_disc[:,0,:][:,None,:],axis=1)
            Vt_disc_closed                   = np.append(Vt_disc,Vt_disc[:,0,:][:,None,:],axis=1)
            Va_ind_disc_closed               = np.append(Va_ind_disc,Va_ind_disc[:,0,:][:,None,:],axis=1)
            Vt_ind_disc_closed               = np.append(Vt_ind_disc,Vt_ind_disc[:,0,:][:,None,:],axis=1)
            
            # Generate functions for interpolation
            blade_T_distribution_function = interp1d(psi_closed,blade_T_distribution_disc_closed,axis=1)
            blade_Q_distribution_function = interp1d(psi_closed,blade_Q_distribution_disc_closed,axis=1)
            blade_Gamma_function          = interp1d(psi_closed,blade_Gamma_disc_closed,axis=1)
            Va_function                   = interp1d(psi_closed,Va_disc_closed,axis=1)
            Vt_function                   = interp1d(psi_closed,Vt_disc_closed,axis=1)
            Va_ind_function               = interp1d(psi_closed,Va_ind_disc_closed,axis=1)
            Vt_ind_function               = interp1d(psi_closed,Vt_ind_disc_closed,axis=1)
            
            # Interpolate to get distributions at each blade for each aximuthal offset. SHAPES = (cpts, Nsteps, B, Nr)
            blade_T_distribution_blades = blade_T_distribution_function(blade_offsets)
            blade_Q_distribution_blades = blade_Q_distribution_function(blade_offsets)
            Gamma_blades                = blade_Gamma_function(blade_offsets)
            Va_blades                   = Va_function(blade_offsets)
            Vt_blades                   = Vt_function(blade_offsets)
            Vt_ind_blades               = Va_ind_function(blade_offsets)
            Va_ind_blades               = Vt_ind_function(blade_offsets)       
            
            
        else:
            # Rotor is experiencing uniform axial flow, all blades experience same instantaneous force distribution
            blade_T_distribution_blades = np.repeat(blade_T_distribution[:,None,:],B,axis=1)                # Shape: [cpts,B,Nr]
            blade_T_distribution_blades = np.repeat(blade_T_distribution_blades[:,None,:,:],Nsteps,axis=1)  # Shape: [cpts,Nsteps,B,Nr]
            blade_Q_distribution_blades = np.repeat(blade_Q_distribution[:,None,:],B,axis=1)                # Shape: [cpts,B,Nr]
            blade_Q_distribution_blades = np.repeat(blade_Q_distribution_blades[:,None,:,:],Nsteps,axis=1)  # Shape: [cpts,Nsteps,B,Nr]
            Va_blades                   = np.repeat(Wa[:,None,:],B,axis=1)                                  # Shape: [cpts,B,Nr]
            Va_blades                   = np.repeat(Va_blades[:,None,:,:],Nsteps,axis=1)                    # Shape: [cpts,Nsteps,B,Nr]
            Vt_blades                   = np.repeat(Wt[:,None,:],B,axis=1)                                  # Shape: [cpts,B,Nr]
            Vt_blades                   = np.repeat(Vt_blades[:,None,:,:],Nsteps,axis=1)                    # Shape: [cpts,Nsteps,B,Nr]
            Va_ind_blades               = np.repeat(va[:,None,:],B,axis=1)                                  # Shape: [cpts,B,Nr]
            Va_ind_blades               = np.repeat(Va_ind_blades[:,None,:,:],Nsteps,axis=1)                # Shape: [cpts,Nsteps,B,Nr]
            Vt_ind_blades               = np.repeat(vt[:,None,:],B,axis=1)                                  # Shape: [cpts,B,Nr]  
            Vt_ind_blades               = np.repeat(Vt_ind_blades[:,None,:,:],Nsteps,axis=1)                # Shape: [cpts,Nsteps,B,Nr]
            Gamma_blades                = np.repeat(Gamma[:,None,:],B,axis=1)                               # Shape: [cpts,B,Nr]   
            Gamma_blades                = np.repeat(Gamma_blades[:,None,:,:],Nsteps,axis=1)                 # Shape: [cpts,Nsteps,B,Nr]        
            
            # disc results
            blade_T_distribution_disc = np.repeat(blade_T_distribution[:,None,:],Na,axis=1)  # Shape: [cpts,Na,Nr]
            blade_Q_distribution_disc = np.repeat(blade_Q_distribution[:,None,:],Na,axis=1)  # Shape: [cpts,Na,Nr]
            Va_disc                   = np.repeat(Wa[:,None,:], Na, axis=1)                  # Shape: [cpts,Na,Nr]
            Vt_disc                   = np.repeat(Wt[:,None,:], Na, axis=1)                  # Shape: [cpts,Na,Nr]
            Va_ind_disc               = np.repeat(va[:,None,:], Na, axis=1)                  # Shape: [cpts,Na,Nr]
            Vt_ind_disc               = np.repeat(vt[:,None,:], Na, axis=1)                  # Shape: [cpts,Na,Nr]
            
        
        # Instantaneous thrust and torque derivatives on each blade
        blade_dT_dr = np.diff(blade_T_distribution_blades)/np.diff(r_1d)
        blade_dQ_dr = np.diff(blade_Q_distribution_blades)/np.diff(r_1d)     
        
        # Instantaneous forces produced by this rotor 
        thrust  = np.atleast_2d( np.sum(blade_T_distribution_blades,axis=(2,3)))
        torque  = np.atleast_2d( np.sum(blade_Q_distribution_blades,axis=(2,3)))             
        power   = omega*torque   
        
        # calculate instantaneous coefficients 
        D        = 2*R 
        Cq       = torque/(rho_0*(n*n)*(D*D*D*D*D)) 
        Ct       = thrust/(rho_0*(n*n)*(D*D*D*D))
        Cp       = power/(rho_0*(n*n*n)*(D*D*D*D*D))
        etap     = V*thrust/power 

        # prevent things from breaking 
        Cq[Cq<0]                                               = 0.  
        Ct[Ct<0]                                               = 0.  
        Cp[Cp<0]                                               = 0.  
        thrust[conditions.propulsion.throttle[:,0] <=0.0]      = 0.0
        power[conditions.propulsion.throttle[:,0]  <=0.0]      = 0.0 
        torque[conditions.propulsion.throttle[:,0]  <=0.0]     = 0.0
        thrust[omega[:,0]<0.0]                                 = -thrust[omega[:,0]<0.0]  
        thrust[omega[:,0]==0.0]                                = 0.0
        power[omega[:,0]==0.0]                                 = 0.0
        torque[omega[:,0]==0.0]                                = 0.0
        Ct[omega[:,0]==0.0]                                    = 0.0
        Cp[omega[:,0]==0.0]                                    = 0.0 
        etap[omega[:,0]==0.0]                                  = 0.0 
        
        # assign efficiency to network
        conditions.propulsion.etap = etap   
        
        # store data
        self.azimuthal_distribution                   = psi  
        results_conditions                            = Data     
        outputs                                       = results_conditions( 
                    # Discretization parameters
                    number_radial_stations            = Nr,
                    number_azimuthal_stations         = Na,   
                    disc_radial_distribution          = r_dim_2d,  
                    disc_azimuthal_distribution       = psi_2d, 
                    
                    # Simulation settings
                    thrust_angle                      = theta,
                    velocity                          = Vv,      
                    omega                             = omega, 
                    
                    # Instantaneous velocities at each rotor blade
                    blade_tangential_induced_velocity = Vt_ind_blades, 
                    blade_axial_induced_velocity      = Va_ind_blades,  
                    blade_tangential_velocity         = Vt_blades, 
                    blade_axial_velocity              = Va_blades, 
                    
                    # Velocities on the rotor disc
                    disc_tangential_induced_velocity  = Vt_ind_disc, 
                    disc_axial_induced_velocity       = Va_ind_disc,  
                    disc_tangential_velocity          = Vt_disc, 
                    disc_axial_velocity               = Va_disc, 
                    
                    # Force distributions around the full rotor disc
                    disc_thrust_distribution          = blade_T_distribution_disc, 
                    disc_torque_distribution          = blade_Q_distribution_disc, 
                    
                    # Instantaneous force and gradient distributions along each rotor blade
                    blade_thrust_distribution         = blade_T_distribution_blades, 
                    blade_torque_distribution         = blade_Q_distribution_blades,   
                    blade_dT_dr                       = blade_dT_dr, 
                    blade_dQ_dr                       = blade_dQ_dr,
                    blade_circulation                 = Gamma_blades, 
                    
                    # Instantaneous forces and coefficients 
                    thrust_coefficient                = Ct,    
                    torque_coefficient                = Cq,  
                    power_coefficient                 = Cp,  
                    thrust                            = thrust, 
                    torque                            = torque,
                    power                             = power,
                    propulsive_efficiency             = etap,
                    
                    converged_inflow_ratio            = lamdaw,
                    disc_effective_angle_of_attack    = alpha,
            ) 
    
        return thrust, torque, power, Cp, outputs , etap


def compute_aerodynamic_forces(a_loc, a_geo, cl_sur, cd_sur, ctrl_pts, Nr, Na, Re, Ma, alpha, tc, use_2d_analysis):
    """
    Cl, Cdval = compute_aerodynamic_forces(  a_loc, 
                                             a_geo, 
                                             cl_sur, 
                                             cd_sur, 
                                             ctrl_pts, 
                                             Nr, 
                                             Na, 
                                             Re, 
                                             Ma, 
                                             alpha, 
                                             tc, 
                                             nonuniform_freestream )
                                             
    Computes the aerodynamic forces at sectional blade locations. If airfoil 
    geometry and locations are specified, the forces are computed using the 
    airfoil polar lift and drag surrogates, accounting for the local Reynolds 
    number and local angle of attack. 
    
    If the airfoils are not specified, an approximation is used.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    a_loc                      Locations of specified airfoils                 [-]
    a_geo                      Geometry of specified airfoil                   [-]
    cl_sur                     Lift Coefficient Surrogates                     [-]
    cd_sur                     Drag Coefficient Surrogates                     [-]
    ctrl_pts                   Number of control points                        [-]
    Nr                         Number of radial blade sections                 [-]
    Na                         Number of azimuthal blade stations              [-]
    Re                         Local Reynolds numbers                          [-]
    Ma                         Local Mach number                               [-]
    alpha                      Local angles of attack                          [radians]
    tc                         Thickness to chord                              [-]
    use_2d_analysis            Specifies 2d disc vs. 1d single angle analysis  [Boolean]
                                                     
                                                     
    Outputs:                                          
    Cl                       Lift Coefficients                         [-]                               
    Cdval                    Drag Coefficients  (before scaling)       [-]
    """        
    # If propeller airfoils are defined, use airfoil surrogate 
    if a_loc != None:
        # Compute blade Cl and Cd distribution from the airfoil data  
        dim_sur = len(cl_sur)   
        if use_2d_analysis:
            # return the 2D Cl and CDval of shape (ctrl_pts, Na, Nr)
            Cl      = np.zeros((ctrl_pts,Na,Nr))              
            Cdval   = np.zeros((ctrl_pts,Na,Nr))
            for jj in range(dim_sur):                 
                Cl_af           = cl_sur[a_geo[jj]](Re,alpha,grid=False)  
                Cdval_af        = cd_sur[a_geo[jj]](Re,alpha,grid=False)  
                locs            = np.where(np.array(a_loc) == jj )
                Cl[:,:,locs]    = Cl_af[:,:,locs]
                Cdval[:,:,locs] = Cdval_af[:,:,locs]          
        else:
            # return the 1D Cl and CDval of shape (ctrl_pts, Nr)
            Cl      = np.zeros((ctrl_pts,Nr))              
            Cdval   = np.zeros((ctrl_pts,Nr))  
            
            for jj in range(dim_sur):                 
                Cl_af         = cl_sur[a_geo[jj]](Re,alpha,grid=False)  
                Cdval_af      = cd_sur[a_geo[jj]](Re,alpha,grid=False)  
                locs          = np.where(np.array(a_loc) == jj )
                Cl[:,locs]    = Cl_af[:,locs]
                Cdval[:,locs] = Cdval_af[:,locs]                   
    else:
        # Estimate Cl max 
        Cl_max_ref = -0.0009*tc**3 + 0.0217*tc**2 - 0.0442*tc + 0.7005
        Re_ref     = 9.*10**6      
        Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1
        
        # If not airfoil polar provided, use 2*pi as lift curve slope
        Cl = 2.*np.pi*alpha
    
        # By 90 deg, it's totally stalled.
        Cl[Cl>Cl1maxp]  = Cl1maxp[Cl>Cl1maxp] # This line of code is what changed the regression testing
        Cl[alpha>=np.pi/2] = 0.
        
        # Scale for Mach, this is Karmen_Tsien
        Cl[Ma[:,:]<1.] = Cl[Ma[:,:]<1.]/((1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5+((Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])/(1+(1-Ma[Ma[:,:]<1.]*Ma[Ma[:,:]<1.])**0.5))*Cl[Ma<1.]/2)
        
        # If the blade segments are supersonic, don't scale
        Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.]  
        
        #This is an atrocious fit of DAE51 data at RE=50k for Cd
        Cdval = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
        Cdval[alpha>=np.pi/2] = 2.    
        
    
    # prevent zero Cl to keep Cd/Cl from breaking in bemt  
    Cl[Cl==0] = 1e-6
        
    return Cl, Cdval