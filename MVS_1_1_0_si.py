import numpy as np

def mvs_si_1_1_0(coeff, bias_data):
    version = 1.10  # Model version

    # fitted coefficients
    Rs0 = coeff[0]  # Access region resistance for s terminal [Ohms-micron]
    Rd0 = Rs0  # Access region resistance for d terminal [Ohms-micron] {Generally Rs0=Rd0 for symmetric source and drain}
    delta = coeff[2]  # Drain induced barrier lowering (DIBL) [V/V]
    n0 = coeff[3]  # Subthreshold swing factor [unit-less] {typically between 1.0 and 2.0}
    nd = coeff[4]  # Punch-through factor [1/V]
    vxo = coeff[6] * 1e7  # Virtual-source injection velocity [cm/s]
    mu = coeff[7]  # low field mobility [cm^2/Vs]
    Vt0 = coeff[8]  # Threshold voltage [V]

    # input parameters known and not fitted.
    global input_parms

    type = input_parms[0]  # type of transistor. nFET type=1; pFET type=-1
    W = input_parms[1]  # Transistor width [cm]
    Lgdr = input_parms[2]  # Physical gate length [cm]. This is the designed gate length for litho printing.
    dLg = input_parms[3]  # Overlap length including both source and drain sides [cm].
    gamma = input_parms[5]  # Body-factor [sqrt(V)]
    phib = input_parms[6]  # ~2*phif [V]
    Cg = input_parms[7]  # Gate-to-channel areal capacitance at the virtual source [F/cm^2]
    Cif = input_parms[8]  # Inner-fringing capacitance [F/cm]
    Cof = input_parms[9]  # Outer-fringing capacitance [F/cm]
    etov = input_parms[10]  # Equivalent thickness of dielectric at S/D-G overlap [cm]
    mc = input_parms[11]  # Effective mass of carriers relative to m0 [unitless]
    Tjun = input_parms[12]  # Junction temperature [K].
    beta = input_parms[13]  # Saturation factor. Typ. nFET=1.8, pFET=1.6
    alpha = input_parms[14]  # Empirical parameter associated with threshold voltage shift between strong and weak inversion.
    CTM_select = input_parms[15]  # Parameter to select charge-transport model
    # if CTM_select = 1, then classic DD-NVSAT
    # model is used; for CTM_select other than
    # 1,blended DD-NVSAT and ballistic charge
    # transport model is used.

    zeta = input_parms[16]  # Energy-transfer factor that lies between zero and unity.
    CC = 0 * 3e-13  # Fitting parameter to adjust Vg-dependent inner fringe capacitances {not used in this version.}

    me = 9.1e-31 * mc  # Effective mass [Kg] invoked for ballistic charges
    qe = 1.602e-19  # Elementary charge [Col.]
    kB = 8.617e-5  # Boltzmann constant [eV/K]
    Cofs = 0 * (0.345e-12 / etov) * dLg / 2 + Cof  # s-terminal outer fringing cap [F/cm]
    Cofd = 0 * (0.345e-12 / etov) * dLg / 2 + Cof  # d-terminal outer fringing cap [F/cm]
    Leff = Lgdr - dLg  # Effective channel length [cm]

    # Initialize output arrays
    Idlog = np.zeros(len(bias_data))
    Id = np.zeros(len(bias_data))
    Qs = np.zeros(len(bias_data))
    Qd = np.zeros(len(bias_data))
    Qg = np.zeros(len(bias_data))
    Vdsi_out = np.zeros(len(bias_data))

    for len_bias in range(len(bias_data)):
        Vd_pre = bias_data[len_bias, 0]
        Vg_pre = bias_data[len_bias, 1]
        Vb_pre = bias_data[len_bias, 2]
        Vs_pre = bias_data[len_bias, 3]
        dir = type * np.sign(Vd_pre - Vs_pre)

        Vds = np.abs(Vd_pre - Vs_pre)
        Vgs = np.max(type * (Vg_pre - Vs_pre), type * (Vg_pre - Vd_pre))
        Vbs = np.max(type * (Vb_pre - Vs_pre), type * (Vb_pre - Vd_pre))

        Vt0bs = Vt0 + gamma * (np.sqrt(np.abs(phib - Vbs)) - np.sqrt(phib))

        # Denormalize access resistances and allocate them the "source" and "drain" according to current flow
        Rs = 1e-4 / W * (Rs0 * (1 + dir) + Rd0 * (1 - dir)) / 2
        Rd = 1e-4 / W * (Rd0 * (1 + dir) + Rs0 * (1 - dir)) / 2

        n = n0 + nd * Vds
        aphit = alpha * phit
        nphit = n * phit
        Qref = Cg * nphit

        # Initial values for current calculation
        FF = 1 / (1 + np.exp((Vgs - (Vt0bs - Vds * delta - 0.5 * aphit)) / (aphit)))
        Qinv_corr = Qref * np.log(1 + np.exp((Vgs - (Vt0bs - Vds * delta - FF * aphit)) / (nphit)))
        Qinv = Qref * np.log(1 + np.exp((Vgs - Vt0bs) / (nphit)))
        Rt = Rs + Rd + (Lgdr - dLg) / (W * Qinv * mu)
        vx0 = vxo
        Vdsats = W * Qinv * vx0 * Rt
        Vdsat = Vdsats * (1 - np.exp(-Qinv / Qref)) + phit * np.exp(-Qinv / Qref)
        Fsat = (1 - np.exp(-2 * Vds / Vdsat)) / (1 + np.exp(-2 * Vds / Vdsat))
        Idx = W * Fsat * Qinv_corr * vx0
        Idxx = 1e-15
        dvg = Idx * Rs
        dvd = Idx * Rd
        count = 1

        # Current calculation loop
        while np.max(np.abs((Idx - Idxx) / Idx)) > 1e-10:
            count += 1
            if count > 500:
                break
            Idxx = Idx
            dvg = (Idx * Rs + dvg) / 2
            dvd = (Idx * Rd + dvd) / 2
            dvds = dvg + dvd  # total drop from source to drain

            Vdsi = Vds - dvds
            Vgsi = Vgs - dvg
            Vbsi = Vbs - dvg

            Vsint = Vs + Idx * (Rs0 * 1e-4 / W) * dir
            Vdint = Vd_pre - Idx * (Rd0 * 1e-4 / W) * dir
            Vgsraw = type * (Vg_pre - Vsint)
            Vgdraw = type * (Vg_pre - Vdint)

            # Correct Vgsi and Vbsi
            Vcorr = (1 + 2.0 * delta) * (aphit / 2.0) * np.exp((-Vdsi) / (aphit))
            Vgscorr = Vgs + Vcorr - dvg
            Vbscorr = Vbs + Vcorr - dvg

            Vt0bs = Vt0 + gamma * (np.sqrt(np.abs(phib - Vbscorr)) - np.sqrt(phib))
            Vt0bs0 = Vt0 + gamma * (np.sqrt(np.abs(phib - Vbsi)) - np.sqrt(phib))

            Vtp = Vt0bs - Vdsi * delta - 0.5 * aphit
            Vtp0 = Vt0bs0 - Vdsi * delta - 0.5 * aphit

            eVg = np.exp((Vgscorr - Vtp) / (aphit))
            FF = 1 / (1 + eVg)
            eVg0 = np.exp((Vgsi - Vtp0) / (aphit))
            FF0 = 1 / (1 + eVg0)

            n = n0 + np.abs(nd * Vdsi)
            nphit = n * phit
            Qref = Cg * nphit
            eta = (Vgscorr - (Vt0bs - Vdsi * delta - FF * aphit)) / (nphit)
            Qinv_corr = Qref * np.log(1 + np.exp(eta))
            eta0 = (Vgsi - (Vt0bs0 - Vdsi * delta - FF * aphit)) / (nphit)
            Qinv = Qref * np.log(1 + np.exp(eta0))

            vx0 = vxo
            Vdsats = vx0 * Leff / mu
            Vdsat = Vdsats * (1 - FF) + phit * FF
            Fsat = (np.abs(Vdsi) / Vdsat) / ((1 + (np.abs(Vdsi) / Vdsat) ** beta) ** (1 / beta))
            v = vx0 * Fsat
            Idx = (W * Qinv_corr * v + 1 * Idxx) / 2

        # Current, positive into terminal y
        Id[len_bias] = type * dir * Idx  # in A
        Idlog[len_bias] = np.log10(Id[len_bias])
        Vdsi_out[len_bias] = Vdsi

        # BEGIN CHARGE MODEL
        Vgt = Qinv / Cg

        # Approximate solution for psis in weak inversion
        psis = phib + (1 - gamma) / (1 + gamma) * phit * (1 + np.log(np.log(1 + np.exp(eta0))))
        a = 1 + gamma / (2 * np.sqrt(psis - Vbsi))  # body factor
        Vgta = Vgt / a  # Vdsat in strong inversion
        Vdsatq = np.sqrt(FF0 * (alpha * phit) ** 2 + (Vgta) ** 2)  # Vdsat approx. to extend to weak inversion

        # Modified Fsat for calculation of charge partitioning (DD-NVSAT)
        betaq = beta
        Fsatq = (np.abs(Vdsi) / Vdsatq) / ((1 + (np.abs(Vdsi) / Vdsatq) ** betaq) ** (1 / betaq))
        x = 1 - Fsatq

        # From L. Wei (DD-NVSAT model)
        A = (1 - x) ** 2 / (12 * (1 - (1 - x) / 2))
        B = (5 - 2 * (1 - x)) / (10 - (1 - x) / 2)
        qsc = Qinv * (0.5 - (1 - x) / 6 + A * (1 - B))
        qdc = Qinv * (0.5 - (1 - x) / 3 + A * B)
        qi = qsc + qdc

        # Calculation of "ballistic" channel charge partitioning factors, qsb and qdb
        if Vds < 1e-3:
            kq2 = 2 * qe / me * (zeta * Vdsi) / (vx0 * vx0) * 1e4
            kq4 = kq2 * kq2
            qsb = Qinv * (0.5 - kq2 / 12.0 + kq4 / 32.0)
            qdb = Qinv * (0.5 - kq2 / 6.0 + 3 * kq4 / 32.0)
        else:
            kq = np.sqrt(2 * qe / me * (zeta * Vdsi)) / vx0 * 1e2  # 1e2 to convert cm/s to m/s. kq is unitless
            kq2 = kq ** 2
            qsb = Qinv * (4 * (kq2 + 1) * np.sqrt(kq2 + 1) - (6 * kq2 + 4)) / (3 * kq2 * kq2)
            qdb = Qinv * (2 * (kq2 - 2) * np.sqrt(kq2 + 1) + 4) / (3 * kq2 * kq2)

        # Flag for classic or ballistic charge partitioning
        if CTM_select == 1:  # classic DD-NVSAT
            qs = qsc
            qd = qdc
        else:  # ballistic blended with classic D/D
            Fsatq2 = Fsatq ** 2
            qs = qsc * (1 - Fsatq2) + qsb * Fsatq2
            qd = qdc * (1 - Fsatq2) + qdb * Fsatq2

        # Body charge based on approximate surface potential (psis) calculation
        Qb[len_bias] = -type * W * Leff * (Cg * gamma * np.sqrt(psis - Vbsi) + (a - 1) / a * Qinv * (1 - qi))

        # Inversion charge partitioning to terminals s and d accounting for source drain reversal
        Qinvs = type * Leff * ((1 + dir) * qs + (1 - dir) * qd) / 2
        Qinvd = type * Leff * ((1 - dir) * qs + (1 + dir) * qd) / 2

        # Overlap and outer fringe S and D to G charges
        Qxov = Cofs * (Vg_pre - Vsint)
        Qyov = Cofd * (Vg_pre - Vdint)

        # Inner fringing S and D to G charges
        Vt0x = Vt0 + gamma * (np.sqrt(phib - type * (Vb_pre - Vsint)) - np.sqrt(phib))
        Vt0y = Vt0 + gamma * (np.sqrt(phib - type * (Vb_pre - Vdint)) - np.sqrt(phib))
        Fs = 1 + np.exp((Vgsraw - (Vt0x - Vdsi * delta * Fsat) + aphit * 0.5) / (1.1 * nphit))
        Fd = 1 + np.exp((Vgdraw - (Vt0y - Vdsi * delta * Fsat) + aphit * 0.5) / (1.1 * nphit))
        FFx = Vgsraw - nphit * np.log(Fs)
        FFy = Vgdraw - nphit * np.log(Fd)
        Qxif = (type * Cif + CC * Vgsraw) * FFx
        Qyif = (type * Cif + CC * Vgdraw) * FFy

        # Total charge at internal terminals x and y
        Qs[len_bias] = -W * (Qinvs + Qxov + Qxif)
        Qd[len_bias] = -W * (Qinvd + Qyov + Qyif)

        # Final charge balance
        Qg[len_bias] = -(Qs[len_bias] + Qd[len_bias] + Qb[len_bias])

    return Idlog, Id, Qs, Qd, Qg, Vdsi_out
