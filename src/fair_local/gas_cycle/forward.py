"""Module for the forward (emissions to concentration) model."""

import numpy as np

from ..constants import GASBOX_AXIS


def step_concentration(
    emissions,
    gasboxes_old,
    airborne_emissions_old,
    alpha_lifetime,
    baseline_concentration,
    baseline_emissions,
    concentration_per_emission,
    lifetime,
    partition_fraction,
    timestep,
):
    """Calculate concentrations from emissions of any greenhouse gas.

    Parameters
    ----------
    emissions : np.ndarray
        emissions rate (emissions unit per year) in timestep.
    gas_boxes_old : np.ndarray
        the greenhouse gas atmospheric burden in each lifetime box at the end of
        the previous timestep.
    airborne_emissions_old : np.ndarray
        The total airborne emissions at the beginning of the timestep. This is
        the concentrations above the pre-industrial control. It is also the sum
        of gas_boxes_old if this is an array.
    alpha_lifetime : np.ndarray
        scaling factor for `lifetime`. Necessary where there is a state-
        dependent feedback.
    baseline_concentration : np.ndarray
        original (possibly pre-industrial) concentration of gas(es) in question.
    baseline_emissions : np.ndarray or float
        original (possibly pre-industrial) emissions of gas(es) in question.
    concentration_per_emission : np.ndarray
        how much atmospheric concentrations grow (e.g. in ppm) per unit (e.g.
        GtCO2) emission.
    lifetime : np.ndarray
        atmospheric burden lifetime of greenhouse gas (yr). For multiple
        lifetimes gases, it is the lifetime of each box.
    partition_fraction : np.ndarray
        the partition fraction of emissions into each gas box. If array, the
        entries should be individually non-negative and sum to one.
    timestep : float
        emissions timestep in years.

    Notes
    -----
    Emissions are given in time intervals and concentrations are also reported
    on the same time intervals: the airborne_emissions values are on time
    boundaries and these are averaged before being returned.

    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.

    Returns
    -------
    concentration_out : np.ndarray
        greenhouse gas concentrations at the centre of the timestep.
    gas_boxes_new : np.ndarray
        the greenhouse gas atmospheric burden in each lifetime box at the end of
        the timestep.
    airborne_emissions_new : np.ndarray
        airborne emissions (concentrations above pre-industrial control level)
        at the end of the timestep.
    """
    decay_rate = timestep / (alpha_lifetime * lifetime)
    decay_factor = np.exp(-decay_rate)

    # additions and removals
    gasboxes_new = (
        partition_fraction
        * (emissions - baseline_emissions)
        * 1
        / decay_rate
        * (1 - decay_factor)
        * timestep
        + gasboxes_old * decay_factor
    )

    airborne_emissions_new = np.sum(gasboxes_new, axis=GASBOX_AXIS)
    concentration_out = (
        baseline_concentration + concentration_per_emission * airborne_emissions_new
    )
    
    flux_new = (partition_fraction * (emissions)) - (gasboxes_new - gasboxes_old)

    return concentration_out, gasboxes_new, airborne_emissions_new, flux_new

def step_concentration_gems(
    gems_beta_550,
    gems_Q10_resp,
    gems_kwScalar,
    gems_PsiScalar,
    gems_dPsidb,
    i_timepoint,
    emissions,
    gasboxes_old,
    airborne_emissions_old,
    alpha_lifetime,
    baseline_concentration,
    baseline_emissions,
    concentration_per_emission,
    lifetime,
    partition_fraction,
    timestep,
    temperature,
):
    """Calculate concentrations from emissions of any greenhouse gas.

    Parameters
    ----------
    i_timepoint : float
        year of simulation
    emissions : np.ndarray
        emissions rate (emissions unit per year) in timestep.
    gas_boxes_old : np.ndarray
        the greenhouse gas atmospheric burden in each lifetime box at the end of
        the previous timestep.
    airborne_emissions_old : np.ndarray
        The total airborne emissions at the beginning of the timestep. This is
        the concentrations above the pre-industrial control. It is also the sum
        of gas_boxes_old if this is an array.
    alpha_lifetime : np.ndarray
        scaling factor for `lifetime`. Necessary where there is a state-
        dependent feedback.
    baseline_concentration : np.ndarray
        original (possibly pre-industrial) concentration of gas(es) in question.
    baseline_emissions : np.ndarray or float
        original (possibly pre-industrial) emissions of gas(es) in question.
    concentration_per_emission : np.ndarray
        how much atmospheric concentrations grow (e.g. in ppm) per unit (e.g.
        GtCO2) emission.
    lifetime : np.ndarray
        atmospheric burden lifetime of greenhouse gas (yr). For multiple
        lifetimes gases, it is the lifetime of each box.
    partition_fraction : np.ndarray
        the partition fraction of emissions into each gas box. If array, the
        entries should be individually non-negative and sum to one.
    timestep : float
        emissions timestep in years.

    Notes
    -----
    Emissions are given in time intervals and concentrations are also reported
    on the same time intervals: the airborne_emissions values are on time
    boundaries and these are averaged before being returned.

    Where array input is taken, the arrays always have the dimensions of
    (scenario, species, time, gas_box). Dimensionality can be 1, but we
    retain the singleton dimension in order to preserve clarity of
    calculation and speed.

    Returns
    -------
    concentration_out : np.ndarray
        greenhouse gas concentrations at the centre of the timestep.
    gas_boxes_new : np.ndarray
        the greenhouse gas atmospheric burden in each lifetime box at the end of
        the timestep.
    airborne_emissions_new : np.ndarray
        airborne emissions (concentrations above pre-industrial control level)
        at the end of the timestep.
    """
    
#     Things that need to be added:
    # import swann/deutsch box model:
    import os
    from functools import partial
    from scipy import integrate
    from ..box_model_functions import co2_emissions_gems, calc_pco2, get_matrix_index, carbon_climate_derivs
    from ..box_model_setup import configure, runSD
    

#     configure the Swann Model:
#     PE, PS, PL, PO = configure()
    PE, PS, PL, PO = configure(beta_550 = gems_beta_550,
                               Q10_resp = gems_Q10_resp,
                               kwScalar = gems_kwScalar,
                               PsiScalar = gems_PsiScalar, 
                               dPsidb = gems_dPsidb,
                               patm0 = baseline_concentration)
    emissions = emissions

    
#     give y0:
# prescribe or compute initial size of each reservoir/pool
    if PS['DoOcn'] == 1:
        Nut_o = PO['Peq'] * 1e-3 # initial ocean nutrient concentration in each box (mol/m3)
        Coc_o = np.array([2388, 2351, 2322, 2325, 2120, 2177, 2329]) * 1e-3 # initial ocean DIC (mol/m3)
        Tmp_o = 0*Nut_o # initial temperature anomaly (deg C)

    lt_side = -1* np.matmul(PL['acoef'], PL['krate'])
    rt_side = (PL['bcoef'] * PL['NPP_o']).reshape(9, 1)
    Cla_o = np.linalg.solve(lt_side, rt_side) # initial land carbon (steady state)
    Cat_o = PE['patm0'] # initial atmos. pCO2 (uatm, or ppm)
#     print(PE['patm0'])

    # package initial values into array
    # this makes it easier to identify domains (rows) and pools (columns)
    y0mat = PE['m0'] # start with dummy matrix

    y0mat_shape = y0mat.shape
    y0mat_flat = y0mat.flatten()

    y0mat_flat[PE['Icla']] = Cla_o.flatten()
    y0mat_flat[PE['Iatm']] = Cat_o

    if PS['DoOcn'] == 1:
        y0mat_flat[PE['Itmp']] = Tmp_o.flatten() # put temperature values into matrix
        y0mat_flat[PE['Inut']] = Nut_o.flatten() # ditto ocean nutrient, etc.
        y0mat_flat[PE['Icoc']] = Coc_o.flatten()

    y0mat = y0mat_flat.reshape(y0mat_shape)
    
    
#     integrate forward
    # convert matrix of initial values to a vector (required for ode solver)
    y0=y0mat.flatten()[PE['Ires']].reshape(1, len(PE['Ires']))

    spery = PE['spery']
    emit = emissions/ 44.009 * 12.011

#     read in temperature from FaIR:
    temp = np.squeeze(np.array(temperature))
    
    if i_timepoint == 0:
        gasboxes = y0
        
        T = np.squeeze(gasboxes)[PE['Jtmp']].transpose()
        T = np.zeros_like(np.array([PE['Jtmp']])).transpose()
#         could initialize fair with only 2 layers `f = FAIR(n_layers=2)`, but for now, have 3 and only use 0 and 1
        T[PO['iAA']] = temp[0]
        T[PO['iSA']] = temp[0]
        T[PO['iNA']] = temp[0]
        T[PO['iLL']] = temp[0]
        T[PO['iTC']] = temp[1]
        if len(temp) < 3:
            T[PO['iNADW']] = temp[1]
            T[PO['iAABW']] = temp[1]
        else:
            T[PO['iNADW']] = temp[2]
            T[PO['iAABW']] = temp[2]


        trun = np.arange(0, (5000)*PE['spery'], PE['spery'])
        new_deriv = partial(carbon_climate_derivs, PE=PE, PS=PS, PL=PL, PO=PO, emit = 0, temperature = T)
        sol = integrate.solve_ivp(new_deriv, t_span = [trun[0],trun[-1]], y0=np.squeeze(gasboxes), method='BDF',t_eval=trun)
        y = np.squeeze(sol.y[:,-1])
        
        t_span = [(5000+i_timepoint)*spery,(5000+i_timepoint+1)*spery]
        new_deriv = partial(carbon_climate_derivs, PE=PE, PS=PS, PL=PL, PO=PO, emit = emit, temperature = T)
        sol = integrate.solve_ivp(new_deriv, t_span = t_span, y0=np.squeeze(y), method='BDF',t_eval=np.array([(5000+i_timepoint+1)*spery]))
        y = np.squeeze(sol.y[:,-1])
        
    else:
        T = np.zeros_like(np.array([PE['Jtmp']])).transpose()
#         could initialize fair with only 2 layers `f = FAIR(n_layers=2)`, but for now, have 3 and only use 0 and 1
        T[PO['iAA']] = temp[0]
        T[PO['iSA']] = temp[0]
        T[PO['iNA']] = temp[0]
        T[PO['iLL']] = temp[0]
        T[PO['iTC']] = temp[1]
        T[PO['iNADW']] = temp[2]
        T[PO['iAABW']] = temp[2]

        t_span = [(5001+i_timepoint)*spery,(5001+i_timepoint+1)*spery]
        gasboxes = np.squeeze(np.array(gasboxes_old[...,:]))
#         print(np.shape(gasboxes))
#         print(np.shape(temp))
        gasboxes[PO['Isfc']] = temp[0]
        gasboxes[PO['iTC']] = temp[1]
        gasboxes[PO['iNADW']] = temp[2]
        gasboxes[PO['iAABW']] = temp[2]
        
        new_deriv = partial(carbon_climate_derivs, PE=PE, PS=PS, PL=PL, PO=PO, emit = emit, temperature = T)
        sol = integrate.solve_ivp(new_deriv, t_span = t_span, y0=gasboxes, method='BDF',t_eval=np.array([(5001+i_timepoint+1)*spery]))
        y = np.squeeze(sol.y)

    gasboxes_new = y.flatten()# * 44.009 / 12.011
    concentration_out = y.flatten()[-1] * 1e6 #* PE['ma'] * 12e-15 * 44.009 / 12.011
    airborne_emissions_new = y[-1] * 1e6 *2.12 * 44.009 / 12.011 #* PE['ma'] * 12e-15 * 44.009 / 12.011
    
    return concentration_out, gasboxes_new, airborne_emissions_new
#     return emissions
