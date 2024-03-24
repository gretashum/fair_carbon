#!/bin/env python 

def co2_emissions_gems(yr, emissions, spinup_time = None):
    from scipy.interpolate import interp1d
    import numpy as np
    
    if spinup_time:
        spinup_time = spinup_time
    else:
        spinup_time = np.arange(0, 5000, step=1)
    time = np.arange(0, len(spinup_time)+len(emissions), step=1)
#     yr = time[yr+len(spinup_time)]
    spinup_emit = np.array([0] * len(spinup_time))
    emit = np.concatenate([spinup_emit,emissions])
    
#     ann_emit = emit[int(yr)]

    time = np.insert(time, 0, -1e6) #GEMS
    time = np.append(time, 1e6) #GEMS

# #     emit = [0, emit, emit[-1]]
    emit = np.insert(emit, 0, 0) #GEMS
    emit = np.append(emit, emit[-1]) #GEMS
# GEMS commented


    # FF=interp1(time,emit,yr);         
    #FF = interp1d(time, emit, yr)
#     FF = np.interp(yr, time, emit) #GEMS
    FF_fctn = interp1d(time, emit) #GEMS commented
    FF = FF_fctn(yr) #GEMS commented

#     return(FF)
    return(FF)


def calc_pco2(t, s, ta, c, phg):
    '''
    this function calculates the partial pressure of co2
    '''

    import numpy as np

    pt = 0e-3
    sit = 40.0e-3
    tk = 273.15 + t
    tk100 = tk / 100.0
    tk1002 = tk100**2
    invtk = 1.0 / tk
    dlogtk = np.log(tk)
    ### note this variable has to change names since "is" is inbuilt in python 
    # is    = 19.924*s./(1000.-1.005*s);
    iss = 19.924 * s / (1000. - 1.005 * s)
    # is2   =is.*is;
    iss2 = iss**2
    sqrtis = np.sqrt(iss)
    s2 = s**2
    sqrts = np.sqrt(s)
    s15 = s ** 1.5
    scl = s / 1.80655

    fflocal = (np.exp(-162.8301 + 218.2968 / tk100  + 
               90.9241 * np.log(tk100) - 1.47696 * tk1002 + 
               s * (.025695 - .025225 * tk100 + 
               0.0049867 * tk1002)))

    k0local = (np.exp(93.4517 / tk100 - 60.2409 + 
               23.3585 * np.log(tk100) + 
               s * (0.023517 - 0.023656 * tk100 + 
               0.0047036 * tk1002)))

    k1local = 10**((-1 * (3670.7 * invtk - 
              62.008 + 9.7944 * dlogtk - 
              0.0118 * s + 0.000116 * s2)))

    k2local = 10**(-1 * (1394.7 * invtk + 4.777 - 
              0.0184 * s + 0.000118 * s2))

    kblocal = np.exp((-8966.90 - 2890.53 * sqrts - 77.942 * s + 
                     1.728 * s15 - 0.0996 * s2) * invtk + 
                     (148.0248 + 137.1942 * sqrts + 1.62142 * s) + 
                     (-24.4344 - 25.085 * sqrts - 0.2474 * s) * 
                     dlogtk + 0.053105 *sqrts * tk)

    k1plocal = np.exp(-4576.752 * invtk + 115.525 - 
                      18.453 * dlogtk + 
                     (-106.736 * invtk + 0.69171) * sqrts + 
                     (-0.65643 * invtk - 0.01844) * s)

    k2plocal = np.exp(-8814.715 * invtk + 172.0883 - 
                      27.927 * dlogtk + 
                     (-160.340 * invtk + 1.3566) * sqrts +  
                     (0.37335 * invtk - 0.05778) * s)

    k3plocal = np.exp(-3070.75 * invtk - 18.141 + 
                      (17.27039 * invtk + 2.81197) *
                      sqrts + (-44.99486 * invtk - 0.09984) * s)

    ksilocal = np.exp(-8904.2 * invtk + 117.385 - 
                      19.334 * dlogtk + 
                      (-458.79 * invtk + 3.5913) * sqrtis + 
                      (188.74 * invtk - 1.5998) * iss + 
                      (-12.1652 * invtk + 0.07871) * iss2 +  
                      np.log(1.0 - 0.001005 * s))

    kwlocal = np.exp(-13847.26 * invtk + 148.9652 -
                     23.6521 * dlogtk +
                    (118.67 * invtk - 5.977 + 1.0495 * dlogtk) * 
                    sqrts - 0.01615 * s)

    kslocal = np.exp(-4276.1 * invtk + 141.328 - 
                     23.093 * dlogtk + 
                     (-13856 * invtk + 324.57 - 47.986 * dlogtk) *sqrtis + 
                     (35474 * invtk - 771.54 + 114.723 * dlogtk) *iss - 
                     2698 * invtk * iss**1.5 + 1776 * invtk * iss2 + 
                     np.log(1.0 - 0.001005 * s))

    kflocal = np.exp(1590.2 * invtk - 12.641 + 1.525 * sqrtis + 
                     np.log(1.0 - 0.001005 * s) +
                     np.log(1.0 + (0.1400 / 96.062) * (scl) / kslocal))

    btlocal = 0.000232 * scl/10.811
    stlocal = 0.14 * scl/96.062
    ftlocal = 0.000067 * scl/18.998

    pHlocal = phg
    permil =1.0 / 1024.5
    pt = pt * permil
    sit = sit * permil
    ta = ta * permil
    c = c * permil

    ####################
    ## start iteration ##
    ####################

    phguess = pHlocal
    hguess = 10.0**(-phguess)
    bohg = btlocal*kblocal / (hguess + kblocal)  # boh4- is an alkalinity species
    stuff = (hguess * hguess * hguess 
             + (k1plocal * hguess * hguess)
             + (k1plocal * k2plocal * hguess) 
             + (k1plocal * k2plocal * k3plocal))
    h3po4g = (pt * hguess * hguess * hguess) / stuff
    h2po4g = (pt * k1plocal * hguess * hguess) / stuff
    hpo4g = (pt * k1plocal * k2plocal * hguess) / stuff
    po4g = (pt * k1plocal * k2plocal * k3plocal) / stuff

    siooh3g = sit * ksilocal / (ksilocal + hguess);

    cag = (ta - bohg - (kwlocal / hguess) + hguess 
           - hpo4g - 2.0*po4g + h3po4g - siooh3g)

    gamm  = c / cag
    hnew = (0.5 * (-k1local * (1 - gamm) + np.sqrt((k1local**2) * (1 - gamm)**2 
            +4 * k1local * k2local * (2 * gamm - 1) ) ))

    pHlocal_new = -np.log10(hnew)
    pHlocal = pHlocal_new

    pco2local = (c / fflocal / (1.0 + (k1local / hnew) +  
                 (k1local * k2local / (hnew**2))))
    fflocal = fflocal / permil

    return(pco2local, pHlocal, fflocal)

def get_matrix_index(arr_row_num, arr_col_num, row_ind, col_ind):
    import numpy as np
    pool_indices = []
    element_nums = np.arange(0, 9*5).reshape(arr_col_num, arr_row_num).transpose()
    for ind in range(0, len(row_ind)):
        # print(element_nums[row_ind[ind], col_ind[0][ind]])
        pool_indices.append(element_nums[row_ind[ind], col_ind[0][ind]])
    return(pool_indices)

def carbon_climate_derivs(t, y, PE, PS, PL, PO, emit, temperature):
    '''
    this is the main function for the box model
    '''

    import numpy as np
    from scipy.interpolate import interp1d
    #import seawater as sw
    # added the necessary seawater functions to their own .py module
    from .seawater_functions import dens0, dens, seck, T68conv
    
    Tloc = y[PE['Jtmp']].transpose() #orig
#     T_0 = temperature[0]
#     T_1 = temperature[1]
#     dT = T_1 - T_0
#     temp = np.empty_like(y[PE['Jtmp']].transpose())
#     temp[0:4] = temperature
#     Tloc = temp
#     taking temperature from FaIR
#     Tloc = temperature
#     print(np.shape(y[PE['Jtmp']].transpose()))
#     print(np.shape(Tloc))
    
    Nloc = y[PE['Jnut']].transpose()
    Dloc = y[PE['Jcoc']].transpose()
    Cloc = y[PE['Jcla']]
    patm = y[PE['Jatm']]


    ## special cases for ocean carbon pumps

    # homogenize T,S if no solubility pump (for pCO2 only)
    ############################### NOTE: Need to add T from whatever dict it's coming from ##################################
    if PS['DoOcn'] == 1:
        
        Tsol = PO['T']
        Ssol = PO['S']
        
        if PS['DoOcnSol'] == 0: 
            Ttmp=Tsol.flatten()
            Stmp=Ssol.flatten()
            Tsol[0,PO['Isfc']] = np.sum(Ttmp[PO['Isfc']] * PO['A'][PO['Isfc']]) / np.sum(PO['A'][PO['Isfc']])
            Ssol[0,PO['Isfc']] = np.sum(Stmp[PO['Isfc']] * PO['A'][PO['Isfc']]) / np.sum(PO['A'][PO['Isfc']])

        # homogenize alkalinity if no bio pump
        TAsol = PO['TA']
        if PS['DoOcnBio'] == 0:
            TAsol[PO['Isfc']] = np.sum(PO['TA'][PO['Isfc']] * PO['A'][PO['Isfc']]) / np.sum(PO['A'][PO['Isfc']])

    ## update basic quantities

    # time
    ymod = t / PE['spery'] # year in model time (starting from 0)
    ycal = ymod - PS['yspin'] + 5000 #PS['ypert'] # calendar year (negative means "BCE") # GEMS TODO maybe later want to make ypert an input too configure and just set PS['ypert'] to 5000 by default
    if ycal < 5000: #PS['ypert']:
        doAtm = 0 # hold atmospheric co2 constant to equilibrate
    else:
        doAtm = 1 # allow atmospheric co2 to evolve

    # interp1d example
    # matlab: interp1(x, y, xn, 'linear')
    # python: yn_f2 = interp1d(x[::-1], y[::-1])
    # python: yn_py2 = yn_f2(xn)

    # atmosphere + climate
#     FF = co2_emissions(ycal, PS['escheme']) # fossil fuel co2 emissions (Pg/yr)
#     FF = co2_emissions_gems(ycal, emissions = emissions) # fossil fuel co2 emissions (Pg/yr)
    # [ycal FF]
    FF = emit
    FF = FF * 1e15 / 12 / PE['spery'] # convert to molC/s
    RFco2 = 5.35 * np.log(patm / PE['patm0']) * PS['DoRadCO2'] # radiative forcing from CO2
#     print(ycal)
    RFsto=np.interp(round(ycal),PS['Yint'].transpose(), PS['RFint'].transpose())
    RF = (RFco2 + np.nansum(RFsto)) * doAtm
#     dTbar = np.sum(Tloc[PO['Isfc']] * PO['A'][PO['Isfc']]) / np.sum(PO['A'][PO['Isfc']]) # GMST (ocean surface = earth surface) # GEMS comment
#     dTbar = np.sum(dT[PO['Isfc']] * PO['A'][PO['Isfc']]) / np.sum(PO['A'][PO['Isfc']]) # GMST (ocean surface = earth surface) # GEMS added
    dTbar = np.sum(Tloc[PO['Isfc']] * PO['A'][PO['Isfc']]) / np.sum(PO['A'][PO['Isfc']]) # GMST (ocean surface = earth surface) # GEMS added

    #------ terrestrial
    NPPfac = 1 + np.interp(ycal,PS['Yint'].transpose(), PS['NPPint'].transpose())
    
    NPP = PL['NPP_o'] * NPPfac * (1 + PS['CCC_LC'] * PL['beta_fert'] * np.log(patm / PE['patm0'])) # perturbation NPP
    #krate = np.diag(PL['kbase']) * PL['Q10_resp']**(PS['CCC_LT'] * dTbar / 10)  # scaled turnover rate
    krate = PL['kbase'] * PL['Q10_resp']**(PS['CCC_LT'] * dTbar / 10)  # scaled turnover rate (vector)
    ## create a matrix version of krate with values on the diagonal 
    krate_diag = np.zeros((krate.shape[0], krate.shape[0]))
    krate_diag_row, krate_diag_col = np.diag_indices(krate_diag.shape[0])
    krate_diag[krate_diag_row, krate_diag_col] = np.squeeze(krate) # matrix version
    Rh = np.sum(-np.sum(PL['acoef'],0) * np.transpose(krate) * Cloc) # Heterotrophic respiration
    
    # To get back to PgC for land pools we take Cloc*(land area)*12e-15. This means that Cloc is in mol/km2
    NEE = (NPP - Rh) * PL['Ala'] # total carbon pool tendency (mol/s)
    
    # set fluxes to 0 in ocean-only case
    if PS['DoTer'] == 0:
        NEE = 0
        krate = 0
        NPP = 0
        Rh = 0

    #------ ocean
    if PS['DoOcn'] == 1:
        Qbio = PO['Qup'] + PO['Qrem']
#         pco2loc, pHloc, Ksol = calc_pco2(Tsol + PS['CCC_OT'] * Tloc, Ssol, TAsol, Dloc, PO['pH0']) # CO2 chemistry GEMS
#         pco2loc, pHloc, Ksol = calc_pco2(T_0 + PS['CCC_OT'] * dT, Ssol, TAsol, Dloc, PO['pH0']) # CO2 chemistry # GEMS using raw temperature from FaIR
        pco2loc, pHloc, Ksol = calc_pco2(Tsol + PS['CCC_OT'] * Tloc, Ssol, TAsol, Dloc, PO['pH0']) # CO2 chemistry GEMS
        pco2Cor = patm * PS['CCC_OC'] + PE['patm0'] * (1 - PS['CCC_OC']) # switch for ocean carbon-carbon coupling
        Fgasx = PO['kwi'] * PO['A'] * Ksol * (pco2loc - pco2Cor) # gas exchange rate

        # circulation change
        #rho = sw.dens(PO['S'], PO['T'] + Tloc, PO['T'] * 0).flatten() # density
#         rho = dens(PO['S'], PO['T'] + Tloc, PO['T'] * 0).flatten() # density GEMS
#         rho = dens(PO['S'], T_1, T_0 * 0).flatten() # density # GEMS using raw T from FaIR
        rho = dens(PO['S'], PO['T'] + Tloc, PO['T'] * 0).flatten() # density GEMS
        bbar = PO['rho_o'][6] - PO['rho_o'][2]
        db = (rho[6] - rho[2]) - bbar
        Psi = PO['Psi_o'] * (1 - PS['CCC_OT'] * PO['dPsidb'] * db / bbar)
        
        #------ Compute Tendencies - should have units mol/s
        dNdt = np.matmul(Psi + Qbio, Nloc.transpose()) ######!!!! There is likely a problem with dNdt - need to check with matlab
        dDdt = np.matmul(Psi, Dloc.transpose()) + PO['Rcp'] * np.matmul(Qbio, Nloc.transpose()) - Fgasx / PO['V'].transpose()  #DIC
        
        
    # set fluxes to 0 in land-only case
    if PS['DoOcn'] == 0:
        Fgasx = 0
        Psi = PO['Psi_o'] # this probably gets set somewhere else when the ocn is turned on.. check

    # [ycal/yend]

    #------ Compute Tendencies - should have units mol/s
#     dTdt = np.matmul(Psi,Tloc.transpose()) -((PO['lammbda'] / PO['V']) * Tloc).transpose() + RF / PO['cm'].transpose() ###!!! problem here too? # there's a ten year timescale to get tempreature into the ocean # GEMS comment
#     dTdt = np.matmul(Psi,dT.transpose()) -((PO['lammbda'] / PO['V']) * dT).transpose() + RF / PO['cm'].transpose() ###!!! problem here too? # there's a ten year timescale to get tempreature into the ocean # added by GEMS
    dTdt = np.matmul(Psi,Tloc.transpose()) -((PO['lammbda'] / PO['V']) * Tloc).transpose() + RF / PO['cm'].transpose() ###!!! problem here too? # there's a ten year timescale to get tempreature into the ocean # GEMS comment

    dAdt = (1 / PE['ma']) * (np.sum(Fgasx) - NEE + FF) # mass of the atmosphere over time

    # land tendencies
    dCdt = np.matmul(np.matmul(PL['acoef'],krate_diag),  Cloc.reshape(9, 1)) + NPP * PL['bcoef']
    
    ## matrix of derivatives

    dydtmat = np.copy(PE['m0']) #initialize with a matrix of zeros. Making a copy here to avoid overwriting the values in PE
    if PS['DoOcn'] == 1: 
        dydtmat[0:PE['nb'],1] = dNdt.flatten()
        dydtmat[0:PE['nb'],2] = dDdt.flatten()
        
    dydtmat[0:PE['nb'],0] = dTdt.flatten()
    dydtmat[0, 4] = dAdt * doAtm
    
    if PS['DoTer'] == 1:
        dydtmat[0:PE['np'],3] = dCdt.flatten()

    temporary = np.transpose(dydtmat).flatten()
    dydt=temporary[PE['Ires']]


    return(dydt)

