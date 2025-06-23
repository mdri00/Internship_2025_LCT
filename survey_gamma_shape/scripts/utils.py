#!/usr/bin/env python3

from pathlib import Path, PosixPath
import csv
import numpy as np
from scipy.stats import weibull_min, gamma, probplot
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
    Returns log-likelihood of given Gamma law given the simulated time variates data. Will be called by scipy.optimize.minimize in gamma_fit
    """      
    shape, loc, scale = params
    if shape <= 0 or scale <= 0:
        return np.inf  # invalid
    return -np.sum(gamma.logpdf(data, a=shape, loc=loc, scale=scale))





def fit_gamma(weibull_par : np.ndarray,
              tau_forced : bool,
              save : bool,
              info : str,
              fig_dir : PosixPath,
              type_of_stage_and_shift : str, 
              seed : int = 0):
    """
    From input Weibull parameters, draws 5000 sample time variates and fits a (shifted)Gamma distribution with ML method.

    INPUT:
    Input weibull_par list is in order [scale, shape, shift].
    info is under the form "species_stage" and will be append to saved figure names.
    type_of_stage_and_shift is used to generate another layer of folders inside the fig_dir folder
    
    OUTPUT:
    Generates a goodness-of-fit visualisation plot in figs folder only if argument save is True.
    Returns fitted Gamma parameters via a numpy array giving shape,rate,shift.
    """
    np.random.seed(seed)
    samples = weibull_min.rvs(c=weibull_par[1], scale=weibull_par[0], loc = weibull_par[2], size=5000)

    if tau_forced : 
        bounds = [
            (1e-5, None),   # shape > 0
            (0, 0),         # loc = 0
            (1e-5, None)    # scale > 0
        ]
        constraint = "Shift = 0"
    else:
        bounds = [
            (1e-5, None),   # shape > 0
            (0, 1),         # loc in [0, 1]
            (1e-5, None)    # scale > 0
        ]
        constraint = r"Shift $\in [0;1]$"

    initial_guess = [1.0, 0.0, 1.0]

    result = minimize(
        gamma_nll,
        initial_guess,
        args=(samples,),
        method='Nelder-Mead',
        bounds=bounds,
        tol = 1e-3, # I tested several possibilites and eventually these tol and maximum iterations work well.
        options = {'maxiter': 20000}
    )

    if not result.success:
        raise RuntimeError("Optimization failed:", result.message)
    shape_fit, loc_fit, scale_fit = result.x

    if save:
        # Folder creation
        output_dir = Path(fig_dir) / type_of_stage_and_shift
        output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        output_path = output_dir / f"{info}.png"
        
        # Generation of fit figure for visual validation of goodness-of-fit
        x = np.linspace(0, max(samples), 1000)
        pdf_fit = gamma.pdf(x, a=shape_fit, loc=loc_fit, scale=scale_fit)
    
        fig = plt.figure() #figsize=(6.3, 3.5))
        plt.text(.9,0.2,f"  Shape: {shape_fit:.4f}")
        plt.text(.9,0.5,f"  Loc: {loc_fit:.4f}")
        plt.text(.9,0.8,f"  Scale: {scale_fit:.4f}")
        plt.hist(samples, bins=50, density=True, alpha=0.5, label="Samples from Weibull parameters")
        plt.plot(x, pdf_fit, 'r-', lw=2, label="Fitted Gamma (" + constraint + ")")
        plt.xlim(0,2)
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        

    return np.array([scale_fit, shape_fit, loc_fit])





def fit_all_weibulls(
    input_csv: str,
    output: str,
    raw_data_dir: PosixPath,
    processed_data_dir: PosixPath,
    fig_dir: PosixPath,
    type_of_stage: str,
    seed: int = 0):
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
                    if loc0>=0: # Not fitting negative shifts
                        weibull_pars.append([scale0, shape0, loc0]) # Order required by fit_gamma
                        info_array.append([species, stage])
    weibull_pars = np.array(weibull_pars)
    info_array = np.array(info_array)

    ### Fit of non-shifted gamma distributions
    gamma_pars = np.zeros(weibull_pars.shape)
    print("Beginning non shifted analysis")
    for i,pars in enumerate(weibull_pars):
        print((i,weibull_pars.shape[0]))
        info_temp = f"{info_array[i,0]}_{info_array[i,1]}"

        gamma_pars_temp = fit_gamma(
            weibull_par = pars,
            tau_forced = True,
            save = True,
            info = info_temp,
            fig_dir = fig_dir,
            type_of_stage_and_shift = type_of_stage)       
        gamma_pars[i,:] = gamma_pars_temp

    ### Fit of shifted gamma distributions
    gamma_pars_shifted = np.zeros(weibull_pars.shape)
    print("Beginning shifted analysis")
    for i,pars in enumerate(weibull_pars):
        print((i,weibull_pars.shape[0]))
        info_temp = f"{info_array[i,0]}_{info_array[i,1]}"
        
        gamma_pars_shifted_temp = fit_gamma(
            weibull_par = pars,
            tau_forced = False,
            save = True,
            info = info_temp,
            fig_dir = fig_dir,
            type_of_stage_and_shift = type_of_stage + "_shifted")                  
        gamma_pars_shifted[i,:] = gamma_pars_shifted_temp


    ### Save results
    np.save(processed_data_dir / f"{output}.npy", gamma_pars)
    np.save(processed_data_dir / f"{output}_shifted.npy", gamma_pars_shifted)

    csv_file_path1 = processed_data_dir / f"{output}.csv"
    csv_file_path2 = processed_data_dir / f"{output}_shifted.csv"
    np.savetxt(csv_file_path1, gamma_pars, fmt="%10.4f",delimiter=",")
    np.savetxt(csv_file_path2, gamma_pars_shifted, fmt="%10.4f",delimiter=",")

    # Add header, add two columns giving species and stage
    with open(csv_file_path1, 'r') as file: # non shifted file
        reader = csv.reader(file)
        data = list(reader)
    for i in range(0, len(data)):
        data[i].append(info_array[i,0])
        data[i].append(info_array[i,1])
    with open(csv_file_path1, 'w', newline='') as file:
        ### header
        writer = csv.DictWriter(file, fieldnames = ["shape", "scale", "shift", "species", "stage"])
        writer.writeheader()
        ### data
        writer = csv.writer(file)
        writer.writerows(data)
    
    with open(csv_file_path2, 'r') as file: # shifted file
        reader = csv.reader(file)
        data = list(reader)
    for i in range(0, len(data)):
        data[i].append(info_array[i,0])
        data[i].append(info_array[i,1])
    with open(csv_file_path2, 'w', newline='') as file:
        ### header
        writer = csv.DictWriter(file, fieldnames = ["shape", "scale", "shift", "species", "stage"])
        writer.writeheader()
        ### data
        writer = csv.writer(file)
        writer.writerows(data)

    return (gamma_pars, gamma_pars_shifted)
   

