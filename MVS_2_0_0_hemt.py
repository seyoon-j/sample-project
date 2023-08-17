import numpy as np

def mvs_2_0_0_hemt(coeff, bias_data):
    # Symmetrical Short-Channel MOSFET model for III-V devices.
    # Returns the drain current, Id [A].
    # This model is only valid for Vg >~ Vg(psis=phif) where psis is the surface
    # potential. I.e range of validity is from onset of weak inversion trhough
    # strong inversion. This model uses a new inversion charge model
    # The model is effective-mass-based.
    
    version = 2.00    # Model version
    
    # input parameters known and not fitted.
    type = input_parms[0]  # Type of transistor. nFET type=1; pFET type=-1
    W = input_parms[1]     # Transistor width [m]
    Lgdr = input_parms[2]  # Physical gate length [m]. This is the designed gate length for litho printing.
    dLg = input_parms[3]   # Overlap length including both source and drain sides [m].
    Cins = input_parms[4]  # Gate-channel capacitance [F/m^2]
    Tjun = input_parms[5]  # Junction temperature [K].
    
    # constants
    kB = 8.617e-5          # Boltzmann constant [eV/K]
    qe = 1.602e-19         # Elementary charge [Col.]
    hbar = 6.62e-34/(2*np.pi)  # Reduced Planck's constant [J-s]
    m0 = 9.1e-31           # Free electron mass [Kg]
    eps0 = 8.85e-12        # Permittivity of free space [F/m]
    nq = 1/3               # QM corr. exponent. Theoretical = 1/3.
    
    phit = kB*Tjun         # Thermal voltage [V]
    kT = phit*qe           # Thermal energy [J]
    
    # fitted coefficients
    Vt0 = coeff[0]         # Threshold voltage [V]
    delta = coeff[1]       # Drain-induced barrier lowering (DIBL) [V/V]
    n0 = coeff[2]          # Non-ideality parameter
    
    Rc0 = coeff[3]         # Contact resistance {same for sourace and drain contacts}  [Ohms-meter]
    nacc = coeff[4]        # Access region charge [1/m^2]
    meff = coeff[5]        # Effective mass of carriers at zero kinetic energy [Kg]
    np_mass = coeff[6]*phit  # Non-parabolicity mass increase [1/eV] {m=m_0*(1+np_mass*E)}
    
    mu_eff = coeff[7]      # Long-channel effective mobility [m^2/Vs]
    ksee = coeff[8]        # Parameter for VS velocity 
    
    B = coeff[9]           # Stern QM correction numerator [(C.m)^1/3]
    dqm0 = coeff[10]       # Distance of charge centroid from insulator interface for negligible charge [m] 
    
    eps = coeff[11]        # Relative permittivity of semiconductor channel material  
    
    theta = coeff[12]      # Fitting parameter for blending Lcrit_lin and _sat
    beta = coeff[13]       # Empirical parameter in Fsat
    nd = coeff[14]         # Punch-through factor [1/V]
    
    beta_acc = beta        # Fitting parameter to govern access region resistance
    beta_crit = beta       # Fitting parameter to govern the slope in Lcrit. 
    
    relax0 = coeff[15]     # Successive iteration relaxation coefficient
    
    energy_diff_volt = -(Vt0/n0 + phit*np.log(meff/m0))   # energy_diff is negative of Vt0. 
    # The phit term is used to keep effective Vt constant idependent of choice of meff.
    # The n0 term is used to allow Vt0 to represent shift in Vg independent of n0.
    
    Leff = Lgdr - dLg                    # Effective channel length [m]
    Qacc = qe*nacc                       # Access region charge [C/m^2]
    QB = (B/dqm0)**(1/nq)                # Denominator for Stern correction [C/m^2]
    N2D = meff/(np.pi*hbar**2)*kT        # Effective 2D density of states
    vT_int = np.sqrt(2*kT/(np.pi*meff))  # Thermal velocity in NDG conditions [m/s]
    lambda_int = 2*phit*mu_eff/vT_int    # Back-scattering mean free path [m]
    
    # ########################################################################
    # Computing thermal velocity of carriers in access region 
    eta_acc = np.log(np.exp(Qacc/(qe*N2D)) - 1)                    # Fermi level normalized to phit in access region
    exp_eta_acc = np.exp(Qacc/(qe*N2D)) - 1                         # Exponent of Fermi level normalized to phit in access region
    FDhalfs_acc = FD_half_integral(eta_acc, exp_eta_acc)           # Fermi-Dirac integral of order 1/2 in access region
    extr_coef_acc = FDhalfs_acc/np.log(1 + exp_eta_acc)            # Degeneracy factor for thermal velocity in access region         
    vT_acc = vT_int * extr_coef_acc                                  # Thermal velocity in the acess region 
    # #######################################################################
    
    # ####### model file begins
    # Direction of current flow:
    # dir=+1 when "x" terminal is the source
    # dir=-1 when "y" terminal is the source |
    
    # bias values
    Vd_pre = bias_data[:, 0]
    Vg_pre = bias_data[:, 1]
    Vs_pre = bias_data[:, 2]
    
    Id = np.zeros((len(Vd_pre), 1))
    Qvs = np.zeros((len(Vd_pre), 1))
    Rsource = np.zeros((len(Vd_pre), 1))
    deltaV = np.zeros((len(Vd_pre), 1))
    vsv = np.zeros((len(Vd_pre), 1))
    
    for len_bias in range(len(Vd_pre)):
        Vd = Vd_pre[len_bias]
        Vg = Vg_pre[len_bias]
        Vs = Vs_pre[len_bias]
        dir = type * np.sign(Vd - Vs)
        
        Vds = np.abs(Vd - Vs)
        Vgs = np.max(type * (Vg - Vs), type * (Vg - Vd))
        
        # Initialization of variables
        psi_solx = 0
        psi_solxx = 1
        Qn = 0
        Idx = 0 
        
        Rc = Rc0 / W
        count = 1
               
        # Current calculation loop
        while np.max(np.abs((psi_solx - psi_solxx) / (np.abs(psi_solxx) + np.abs(psi_solx)))) > 1e-12:
            count += 1
            if count > 500:
                break
            
            Idxx = Idx
            psi_solxx = psi_solx
            
            # Non-linear contact resistance at the source and drain terminals
            Idx_sat = W * Qacc * vT_acc      
            Rs = Rc / (1 - (Idx / Idx_sat)**beta_acc)**(1 / beta_acc)    # Non-linear source resistance [Ohms]
            Rd = Rs  # Non-linear drain resistance {assumed equal to source resistance} [Ohms] 
            
            # Internal Vgsi and Vdsi accounting for voltage drops on Rs, Rd
            dvg = Idx * Rs
            dvd = Idx * Rd
            dvds = dvg + dvd             
            Vdsi = Vds - dvds  
            Vgsi = Vgs - dvg
            
            # Effective mass multiplication factor including non-parabolicity
            Es = (energy_diff_volt + psi_solx) / phit
            expEs = np.exp(Es)       
            ffs = 1 / (1 + expEs * 0.6)
            Es_sq = (1 + np.sign(Es)) * Es**2 / 4
            flux_source = np.log(1 + expEs) + np_mass * (ffs * expEs + (1 - ffs) * Es_sq)
            meff_np = flux_source / np.log(1 + expEs)  # "Effective mass" by averaging over all energy states  
     
            Ed = (energy_diff_volt + psi_solx - Vdsi) / phit     
            expEd = np.exp(Ed)
            ffd = 1 / (1 + expEd * 0.6)
            Ed_sq = (1 + np.sign(Ed)) * Ed**2 / 4
            flux_drain = np.log(1 + expEd) + np_mass * (ffd * expEd + (1 - ffd) * Ed_sq)
    
            # Mean free path (lambda) including degeneracy
            FDzero = np.log(1 + expEs)
            dEs = 1e-6
            Esm = Es - dEs
            Esp = Es + dEs
            expEsm = np.exp(Esm)
            expEsp = np.exp(Esp)
            FDhalfsp = FD_half_integral(Esp, expEsp)  # After Blakemore
            FDhalfsm = FD_half_integral(Esm, expEsm)
            FDhalfs = (FDhalfsp + FDhalfsm) / 2
            FDminushalfs = (FDhalfsp - FDhalfsm) / (2 * dEs)
            dgen_lambda = FDzero / FDminushalfs
           
            NP_fac_lambda = np.sqrt(meff_np)   # non-parabolicity factor for lambda in main channel
            lambda_ = lambda_int * dgen_lambda * NP_fac_lambda  # Net mean free path in the channel
            
            # Thermal velocity including degeneracy and non-parabolicity
            extr_coef = FDhalfs / np.log(1 + expEs)  # Degeneracy coefficient for thermal velocity
            NP_fac_velocity = 1 / NP_fac_lambda  # Non-parabolicity factor for thermal velocity in main channel
            vT = vT_int * extr_coef * NP_fac_velocity  # Thermal velocity with degeneracy and non-parab. [m/s]
            
            # Critical length and backscattering [Model-1]
            Lcrit_lin = Leff
            Lcrit_sat = ksee * Leff
            Vdsat2 = theta * phit
            f2 = Vdsi / Vdsat2 / (1 + (Vdsi / Vdsat2)**beta_crit)**(1 / beta_crit)  # Effective Vdsi is used here.
            Lcrit = Lcrit_lin * (1 - f2) + Lcrit_sat * f2
            Tx = lambda_ / (lambda_ + Lcrit)
            
            # Full-saturation VS point injection velocity (max. velocity)
            vx0 = vT * lambda_ / (lambda_ + 2 * ksee * Leff)
            
            # Saturation voltage in main channel
            coef_sat = (2 * flux_source / ((2 - Tx) * flux_source + Tx * flux_drain))**(-1)
            Vdsat = vT * lambda_ / mu_eff * (lambda_ + Leff) / (lambda_ + 2 * ksee * Leff) * coef_sat
            
            # VS point injection velocity (valid from linear to saturation)
            Fsat = (np.abs(Vdsi) / Vdsat) / ((1 + (np.abs(Vdsi) / Vdsat)**beta)**(1 / beta))
            v = vx0 * Fsat
     
            # Charge at VS point
            Qn = -qe * N2D / 2 * ((2 - Tx) * flux_source + Tx * flux_drain)
            
            # QM correction (depth of charge centroid) to Cg
            dqm = B / ((QB + 11 / 32 * np.abs(Qn))**nq)  # Stern correction for centroid distance [m]
            Cstern = eps * eps0 / dqm
            Cgc = Cstern * Cins / (Cstern + Cins)  # Effective gate capacitance [F/m^2]
     
            # Calculate currrent and surface potential
            xx = 1 + ((np.abs(Vdsi) + 1e-15) / (np.abs(Vds) + 1e-15)) / 2  
            relax = relax0 * xx * (extr_coef)**2  # Increase relax for high Rs,d
            
            Idx = (W * np.abs(Qn) * v + 4 * relax * Idxx) / (4 * relax + 1)
            n = n0 + np.abs(nd * Vdsi)
            psi_solx = ((Vgsi + delta * Vdsi + Qn / Cgc) / n + relax * psi_solxx) / (relax + 1)
        
        Id[len_bias, 0] = type * dir * Idx  # in (Amperes)
        
        if count < 3:
            print(count)
        
        Qvs[len_bias, 0] = Qn
        deltaV[len_bias, 0] = Fsat
        vsv[len_bias, 0] = v
        Rsource[len_bias, 0] = Rs
    
    return Id, Qvs, Rsource, deltaV, vsv
