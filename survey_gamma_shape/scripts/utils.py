#!/usr/bin/env python3

from pathlib import Path, PosixPath
import csv
import numpy as np
from scipy.stats import weibull_min, gamma, probplot, wasserstein_distance
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib


from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "font.size" : 15,
    "figure.titlesize": 15,
    "pgf.rcfonts": False    
})



def gamma_nll(params : tuple, 
              data : np.ndarray):
    """
    Computes log-likelihood of the simulated time data from Weibull law, as a function of a given Gamma law that is passed from scipy.stats.optimize.
    Clips time variates from 0 to +inf : otherwise negative-shifted weibull law will yield negative time variates that will produce -inf logproba because the Gamma law is supported on R+
    """      
    shape, loc, scale = params
    data = np.clip(data, 1e-5, np.inf) # Clipping negative time variates : if clipping to exactly 0 LogLikelihood will be -inf
    if shape <= 0 or scale <= 0:
        return np.inf  # invalid
    return -np.sum(gamma.logpdf(data, a=shape, loc=loc, scale=scale))



def fit_gamma(weibull_par : np.ndarray,
              tau_forced : str,
              save : bool,
              info : str,
              fig_dir : PosixPath,
              type_of_stage_and_shift : str, 
              bootstrap_n : int):
    """
    From input Weibull parameters, draws 5000 sample time variates and fits a (shifted)Gamma distribution with ML method.

    INPUT:
    Input weibull_par list is in order [scale, shape, shift].
    tau_forced is a string taking the values : zero (-> shift forced to 0) ; copy (-> shift taken from weibull law) ; positive (-> shift taken in [0;1])
    info is under the form "species_stage" and will be append to saved figure names.
    type_of_stage_and_shift is used to generate another layer of folders inside the fig_dir folder
    bootstrap_n : if 0 then no bootstrap method (just one fit), if > 0 then multiple sampling (i.e 5000 weibull time generation) is made which results in multiple fitted parameters.
    
    OUTPUT:
    Generates a goodness-of-fit visualisation plot in figs folder only if argument save is True.
    Returns average fitted Gamma parameters via a numpy array giving shape,rate,shift.
    Returns standard errors computed from the bootstrapping method.
    Returns average Wasserstein distance between the input Weibull law and the output Gamma law.
    """

    # Bootstrapped estimation of Gamma parameters from 5000 random samples taken from given Weibull law
    Shape_fits = np.zeros(bootstrap_n)
    Scale_fits = np.zeros(bootstrap_n)
    Loc_fits = np.zeros(bootstrap_n)
    
    for bootstrap_i in range(bootstrap_n):
        samples = weibull_min.rvs(c=weibull_par[1], scale=weibull_par[0], loc = weibull_par[2], size=5000)
    
        if tau_forced == "zero" : 
            bounds = [
                (1e-5, None),   # shape > 0
                (0, 0),         # loc = 0
                (1e-5, None)    # scale > 0
            ]
            constraint = "Shift = 0"
            initial_guess = [1.0, 0.0, 1.0]
        elif tau_forced == "positive" :
            bounds = [
                (1e-5, None),   # shape > 0
                (0, 1),        # loc in [0, 1]
                (1e-5, None)    # scale > 0
            ]
            constraint = r"Shift $\in [0;1]$"
            initial_guess = [1.0, 0.0, 1.0]
        elif tau_forced == "copy" :
            bounds = [
                (1e-5, None),   # shape > 0
                (weibull_par[2], weibull_par[2]), # loc copied from Weibull
                (1e-5, None)    # scale > 0
            ]
            constraint = r"Shift copied from Weibull law"
            initial_guess = [1.0, weibull_par[2], 1.0]
        else:
            raise RuntimeError("Tau forced argument not in [zero, unconstrained, copy]")
    
        result = minimize(
            gamma_nll,
            initial_guess,
            args=(samples,),
            method='Nelder-Mead',
            bounds=bounds,
            tol = 1e-3, # I tested several possibilites and eventually these tol and maximum iterations work well.
            options = {'maxiter': 20000} # Same
        )
        
        if not result.success:
            raise RuntimeError("Optimization failed:", result.message)
        shape_fit, loc_fit, scale_fit = result.x
        
        Shape_fits[bootstrap_i] = shape_fit
        Scale_fits[bootstrap_i] = scale_fit
        Loc_fits[bootstrap_i] = loc_fit

    print(Shape_fits)
    print(Loc_fits)

    scale_fit_average = np.mean(Scale_fits)
    shape_fit_average = np.mean(Shape_fits)
    loc_fit_average = np.mean(Loc_fits)

    scale_fit_se = np.std(Scale_fits)/np.sqrt(bootstrap_n)
    shape_fit_se = np.std(Shape_fits)/np.sqrt(bootstrap_n)
    loc_fit_se = np.std(Loc_fits)/np.sqrt(bootstrap_n)

    print(shape_fit_se)
    print(loc_fit_se)

    # Wasserstein distance computation from average parameters 
    support = np.linspace(1e-5,100,1000)
    w_dist = wasserstein_distance(u_values = support, 
                                  v_values = support,
                                  u_weights = weibull_min.pdf(support, c=weibull_par[1], scale=weibull_par[0], loc = weibull_par[2]),
                                  v_weights = gamma.pdf(support, a=shape_fit_average, loc=loc_fit_average, scale=scale_fit_average)
                                 )

    if save:
        # Folder creation
        output_dir = Path(fig_dir) / type_of_stage_and_shift
        output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        output_path = output_dir / f"{info}.png"
        
        # Generation of fit figure for visual validation of goodness-of-fit
        x = np.linspace(0, max(samples), 1000)
        pdf_fit = gamma.pdf(x, a=shape_fit_average, loc=loc_fit_average, scale=scale_fit_average)
    
        fig = plt.figure() #figsize=(6.3, 3.5))
        plt.text(.9,0.2,f"  Shape: {shape_fit_average:.4f}")
        plt.text(.9,0.5,f"  Loc: {loc_fit_average:.4f}")
        plt.text(.1,0.4,f" W-distance : {w_dist:.4f}")
        plt.text(.9,0.8,f"  Scale: {scale_fit_average:.4f}")
        plt.hist(samples, bins=50, density=True, alpha=0.5, label="Samples from Weibull parameters")
        plt.plot(x, pdf_fit, 'r-', lw=2, label="Fitted Gamma (" + constraint + ")")
        plt.xlim(0,2)
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    return (np.array([scale_fit_average, shape_fit_average, loc_fit_average]), np.array([scale_fit_se, shape_fit_se, loc_fit_se]), w_dist)





def fit_all_weibulls(
    input_csv: str,
    output: str,
    raw_data_dir: PosixPath,
    processed_data_dir: PosixPath,
    fig_dir: PosixPath,
    type_of_stage: str,
    bootstrap_n_global):
    """
    Reads csv file containing weibull parameters. Calls fit_gamma to generate the corresponding Gamma distributions with two constraints on tau_forced.

    INPUT:
    raw data CSV file with Weibull parameters taken from literature with order [rate, shape, shit]
    Type of stage is either per_stage (larvae...), whole_dev, or per_substage (larvae instar 1...).

    OUTPUT:
    Generated a csv file similar to the input csv file with the Gamma parameters.
    Also saves the numpy ndarrays with numpy.save for easier loading of fits in Python.
    Returns all fitted Gamma parameters in a tuple of two numpy array. 
    """

    ### Data extraction
    info_array = [] # Contains first columns of csv file giving information about the considered species and maturation stage
    weibull_pars = []
    with open(raw_data_dir / f"{input_csv}.csv", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for i,row in enumerate(reader):
                species = row[0]
                stage = row[1]
                scale0 = np.float32(row[2])
                shape0 = np.float32(row[3])
                loc0 = np.float32(row[4])
                if shape0>0:
                    weibull_pars.append([scale0, shape0, loc0]) # Order required by fit_gamma
                    info_array.append([species, stage])
    weibull_pars = np.array(weibull_pars)
    info_array = np.array(info_array)

    shape1 = weibull_pars.shape[0] # shape of numpy arrays that will be generated and saved an given in output
    shape2 = weibull_pars.shape[1] 
    
    ### Fit of non-shifted gamma distributions
    gamma_av_zero = np.zeros((shape1, shape2))
    gamma_se_zero = np.zeros((shape1, shape2))
    gamma_wd_zero = np.zeros(shape1)
    print("Beginning non shifted analysis")
    for i,pars in enumerate(weibull_pars):
        print((i,weibull_pars.shape[0]))
        info_temp = f"{info_array[i,0]}_{info_array[i,1]}"

        gamma_avs_temp, gamma_ses_temp, w_dist_temp = fit_gamma(
            weibull_par = pars,
            tau_forced = "zero",
            save = True,
            info = info_temp,
            fig_dir = fig_dir,
            type_of_stage_and_shift = type_of_stage + "_zero",
            bootstrap_n = bootstrap_n_global)   
        
        gamma_av_zero[i,:] = gamma_avs_temp
        gamma_se_zero[i,:] = gamma_ses_temp
        gamma_wd_zero[i] = w_dist_temp


    
    ### Fit of non-shifted gamma distributions
    gamma_av_pos = np.zeros((shape1, shape2))
    gamma_se_pos = np.zeros((shape1, shape2))
    gamma_wd_pos = np.zeros(shape1)
    print("Beginning positive shift analysis")
    for i,pars in enumerate(weibull_pars):
        print((i,weibull_pars.shape[0]))
        info_temp = f"{info_array[i,0]}_{info_array[i,1]}"

        gamma_avs_temp, gamma_ses_temp, w_dist_temp = fit_gamma(
            weibull_par = pars,
            tau_forced = "positive",
            save = True,
            info = info_temp,
            fig_dir = fig_dir,
            type_of_stage_and_shift = type_of_stage + "_positive",
            bootstrap_n = bootstrap_n_global)   
        
        gamma_av_pos[i,:] = gamma_avs_temp
        gamma_se_pos[i,:] = gamma_ses_temp
        gamma_wd_pos[i] = w_dist_temp

        
        
    ### Fit of non-shifted gamma distributions
    gamma_av_copy = np.zeros((shape1, shape2))
    gamma_se_copy = np.zeros((shape1, shape2))
    gamma_wd_copy = np.zeros(shape1)
    print("Beginning non shifted analysis")
    for i,pars in enumerate(weibull_pars):
        print((i,weibull_pars.shape[0]))
        info_temp = f"{info_array[i,0]}_{info_array[i,1]}"

        gamma_avs_temp, gamma_ses_temp, w_dist_temp = fit_gamma(
            weibull_par = pars,
            tau_forced = "copy",
            save = True,
            info = info_temp,
            fig_dir = fig_dir,
            type_of_stage_and_shift = type_of_stage + "_copy",
            bootstrap_n = bootstrap_n_global)   
        
        gamma_av_copy[i,:] = gamma_avs_temp
        gamma_se_copy[i,:] = gamma_ses_temp
        gamma_wd_copy[i] = w_dist_temp


    # Concatenate all output in just one table
    fit_results = np.concatenate((gamma_av_zero, gamma_se_zero, gamma_wd_zero[:,np.newaxis], 
                                 gamma_av_pos, gamma_se_pos, gamma_wd_pos[:,np.newaxis], 
                                 gamma_av_copy, gamma_se_copy, gamma_wd_copy[:,np.newaxis]), 
                                 axis = 1)
    
    ### Save fitting outputs 
    #### numpy table
    np.save(processed_data_dir / f"{output}.npy", fit_results)

    #### csv table
    csv_file_path1 = processed_data_dir / f"{output}.csv"
    np.savetxt(csv_file_path1, fit_results, fmt="%10.4f",delimiter=",")

    # Add header, add two columns giving species and stage
    with open(csv_file_path1, 'r') as file: # non shifted file
        reader = csv.reader(file)
        data = list(reader)
    newdatalist = []
    for i in range(0, len(data)):
        newdatalist.append(
            [info_array[i,0], info_array[i,1]]
            + list(weibull_pars[i])
            + list(data[i])
        )
    with open(csv_file_path1, 'w', newline='') as file:
        ### header
        writer = csv.DictWriter(file, fieldnames = ["species", "stage", 
                                                    "weibull_shape", "weibull_scale", "weibull_shift",
                                                    "scale_av_zero", "shape_av_zero", "shift_av_zero", "scale_se_zero", "shape_se_zero", "shift_se_zero", "w-dist_zero", 
                                                    "scale_av_pos", "shape_av_pos", "shift_av_pos", "scale_se_pos", "shape_se_pos", "shift_se_pos", "w-dist_pos",
                                                    "scale_av_copy", "shape_av_copy", "shift_av_copy", "scale_se_copy", "shape_se_copy", "shift_se_copy", "w-dist_copy"
                                                    ])
        writer.writeheader()
        ### data
        writer = csv.writer(file)
        writer.writerows(newdatalist)


    return (fit_results)
   

