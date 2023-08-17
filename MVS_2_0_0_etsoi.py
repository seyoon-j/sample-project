import numpy as np

def mvs_2_0_0_etsoi(coeff, bias_data):
    version = 2.00  # Model version

    # input parameters known and not fitted
    input_parms = [1, 0.1, 0.1, 0.1, 1e-6, 300]  # Example values, replace with actual input_parms

    type = input_parms[0]  # Type of transistor. nFET type=1; pFET type=-1
    W = input_parms[1]  # Transistor width [m]
    Lgdr = input_parms[2]  # Physical gate length [m]
    dLg = input_parms[3]  # Overlap length including both source and drain sides [m]
    Cox = input_parms[4]  # Insulator capacitance [F/m^2] {calibrated from C-V}
    Tjun = input_parms[5]  # Junction temperature [K]

    # constants
    phit = 8.617e-5 * Tjun  # Thermal voltage [V]
    qe = 1.602e-19  # Electron charge [Coul]
    kT = phit * qe  # Thermal energy [Joules]
    hbar = 6.62e-34 / (2 * np.pi)  # Reduced Planck's constant [J-s]
    eps0 = 8.85e-12  # Permittivity of free space [F/m]
    nq = 1 / 3  # QM corr. exponent. Theoretical = 1/3

    # fitted coefficients
    Rs0 = coeff[0]  # Source access resistance [Ohms-meter]
    Rd0 = Rs0  # Assumed symmetric source and drain. If asymmetric, use coeff(2) for Rd0.
    delta = coeff[2]  # Drain induced barrier lowering (DIBL) [V/V]
    n0 = coeff[3]  # Subthreshold swing factor [unit-less] {typically between 1.0 and 2.0}
    nd = coeff[4]  # Punch-through [1/V]
    theta = coeff[5]  # Saturation voltage for critical length
    beta = coeff[6]  # Parameter to control slope of saturation function, Fsat, in transport
    beta_c = coeff[7]  # Parameter to control slope of critical length saturation function
    energy_diff_volt = coeff[8]  # Threshold voltage [V]
    mt = coeff[9]  # Transverse electron mass in Silicon [Kg]
    ml = coeff[10]  # Longitudinal effective mass in Silicon [Kg]
    mu_eff = coeff[11]  # Long-channel effective mobility [m^2/Vs]
    ksee = coeff[12]  # Coefficient for calculating VS injection velocity
    B = coeff[13]  # Stern QM correction numerator
    dqm0 = coeff[14]  # Distance of charge centroid from the interface when channel charge is negligible [m]
    eps = coeff[15]  # Relative permittivity of semiconductor channel material
    nu = coeff[16]  # Relative occupancy of delta2 subbands
    relax0 = coeff[17]  # Successive iteration relaxation coefficient

    Leff = Lgdr - dLg  # Effective channel length [m]
    QB = (B / dqm0) ** (1 / nq)  # QM corr. based on fitted distance of centroid

    # Conductivity and DOS electron masses in delta2 and delta4 sub-bands
    mC_delta2 = 4 * mt
    mD_delta2 = 2 * mt
    mC_delta4 = 4 * (np.sqrt(mt) + np.sqrt(ml)) ** 2
    mD_delta4 = 4 * np.sqrt(mt * ml)

    # Calculate non-degenerate values of thermal velocity and mean free path
    vT_delta2_int = np.sqrt(2 * kT / (np.pi) * mC_delta2 / mD_delta2 ** 2)
    vT_delta4_int = np.sqrt(2 * kT / (np.pi) * mC_delta4 / mD_delta4 ** 2)
    vT_avg_int = nu * vT_delta2_int + (1 - nu) * vT_delta4_int
    lambda_int = 2 * phit * mu_eff / vT_avg_int

    # Initialize output array
    Id = np.zeros(len(bias_data))

    for len_bias in range(len(bias_data)):
        Vd = bias_data[len_bias, 0]
        Vg = bias_data[len_bias, 1]
        Vs = bias_data[len_bias, 2]
        dir = type * np.sign(Vd - Vs)

        Vds = np.abs(Vd - Vs)
        Vgs = np.max(type * (Vg - Vs), type * (Vg - Vd))

        # Initialization of variables
        psi_solx = 0
        psi_solxx = 1
        Qn = 0
        Vdsat = phit
        Idx = 0
        Rs = Rs0 / W
        Rd = Rd0 / W

        dvg = Idx * Rs
        dvd = Idx * Rd
        count = 1

        # Current calculation loop
        while np.max(np.abs((psi_solx - psi_solxx) / (np.abs(psi_solxx) + np.abs(psi_solx)))) > 1e-12:
            count += 1
            if count > 1000:
                break

            Idxx = Idx
            psi_solxx = psi_solx

            lambda_dg = lambda_int
            vT_dg = vT_avg_int

            dvg = (Idx * Rs + dvg * 0) / 1
            dvd = (Idx * Rd + dvd * 0) / 1
            dvds = dvg + dvd
            Vdsi = Vds - dvds
            Vgsi = Vgs - dvg

            Es = (energy_diff_volt + psi_solx) / phit
            expEs = np.exp(Es)
            FDhalfs = FD_half_integral(Es, expEs)  # After Blakemore
            extr_coef = FDhalfs / np.log(1 + expEs)
            Ed = (energy_diff_volt + psi_solx - Vdsi) / phit
            expEd = np.exp(Ed)

            flux_source_delta2 = np.log(1 + expEs)
            flux_source_delta4 = flux_source_delta2 * (1 - nu) / nu
            flux_drain_delta2 = np.log(1 + expEd)
            flux_drain_delta4 = flux_drain_delta2 * (1 - nu) / nu

            flux_source = flux_source_delta2 + flux_source_delta4
            flux_drain = flux_drain_delta2 + flux_drain_delta4

            Lcrit_lin = Leff
            Lcrit_sat = ksee * Leff
            Vdsat_c = theta * phit
            Fsat_c = (np.abs(Vdsi) / Vdsat_c) / ((1 + (np.abs(Vdsi) / Vdsat_c) ** beta_c) ** (1 / beta_c))
            Lcrit = Lcrit_lin * (1 - Fsat_c) + Lcrit_sat * Fsat_c
            Tx = lambda_dg / (lambda_dg + Lcrit)

            coef_sat = (2 * flux_source / ((2 - Tx) * flux_source + Tx * flux_drain)) ** (-1)
            Vdsat = 2 * phit * (lambda_dg + Leff) / (lambda_dg + 2 * ksee * Leff) * coef_sat
            Fsat = (np.abs(Vdsi) / Vdsat) / ((1 + (np.abs(Vdsi) / Vdsat) ** beta) ** (1 / beta))

            vx0 = vT_dg * (lambda_dg / (lambda_dg + 2 * ksee * Leff))
            v = vx0 * Fsat

            Qn = -qe * kT / (2 * np.pi * hbar ** 2) * mD_delta2 * ((2 - Tx) * flux_source + Tx * flux_drain)
            n = n0 + np.abs(nd * Vdsi)
            dqm = B / (QB + 11 / 32 * np.abs(Qn)) ** nq
            Cstern = eps * eps0 / dqm
            Cgc = Cstern * Cox / (Cstern + Cox)

            xx = 1 + ((np.abs(Vdsi) + 1e-15) / (np.abs(Vds) + 1e-15)) / 2
            relax = relax0 * xx * (extr_coef) ** 2
            Idx = (W * np.abs(Qn) * v + 5 * relax * Idxx) / (5 * relax + 1)
            psi_solx = ((Vgsi + delta * Vdsi + Qn / Cgc) / n + 4 * relax * psi_solxx) / (4 * relax + 1)

        Id[len_bias] = type * dir * Idx

    return Id
