# -*- coding: utf-8 -*-
"""
Title: PHYS20161 - Assigment 2 - Z0 boson

This code uses two data files "z_boson_data_1" and "z_boson_data_2" given in 
the assigment to calculate the boson mass, width, and lifetime and their
uncertainties. 

Will only run if input files are of form [centre of mass energy, cross section,
                                           uncertainty in cross section].

1) Validation of files: Search for all csv files in directory and request user 
input for any anomalies. If none, automaticaly continues with the two expected
data files
2) Check files for validation issues, remove anomalies, and create data array
3) Do a gaussian fit to find a generalised estimate of boson_mass and boson_width
4) Remove outliers while simultaneously fitting theoretical cross section to
experimental data from files
5) Perform a minimised chi_squared test by simultaneously varying boson_mass and 
boson_width as part of the fit_parameters
6) Obtain uncertainty by changing the fit_parameteres until a chi_squared
greater than minimum_chi_squared + 1 is obtained
7) Create and format two plots and display relevant information: 
    Plot of data: Cross section against centre of energy
    Plot of chi-squared over m_z and gamma_z
8) Calculate boson lifetime and uncertainty
9) Format uncertainties and print

Last updated: 19/12/23
@author: Lukas Wystemp, UID: 11012465
"""

### IMPORTS ###
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc
from matplotlib.lines import Line2D
from scipy import constants as const

### CONSTANTS ###
BOSON_MASS_EST = 90 # GeVc^-2 # Only used if generalised estimates has runtime error
BOSON_WIDTH_EST = 3 # GeV
PLANCK_CONST = 6.582119569e-25 # GeVs
NUMBER_OF_PARAMETERS = 2 # Because two fit parameters
NATURAL_UNIT_CONVERSION = 0.3894e-3 # b
PARTIAL_WIDTH = 83.91e-3 # GeV

NAME_OF_FIGURE_1 = "Assigment_2_Z0_boson_data" # Customise name of figure of data
NAME_OF_FIGURE_2 = "Assigment_2_contour_plot" # Customise name of figure of chi^2
EXPECTED_FILES = ['z_boson_data_1.csv', 'z_boson_data_2.csv']

### FUNCTIONS ###

def validate_csv_files(expected_files):
    """
    Detects if there are missing files or additionally unexpected files with
    find_csv_files and calls validated_missing_files() or validated_extra_files()
    1) If there are no csv files in directory, exit
    2) If there are any files missing but were expected, asks user if they want
    to proceed anyways
    3) If there are any additional files, asks user if they want to include
    these as well
    
    Takes care of the edge cases where user validates additional files but
    intentionally excludes EXPECED_FILES

    Parameters
    ----------
    expected_files : array
        by default global variable EXPECTED_FILES = ['z_boson_data_1.csv', 
                                                     'z_boson_data_2.csv'].
        This may be amended freely

    Returns
    -------
    validated_files : array
        expected files that are present in directory and additionally found
        files that have been approved by user

    """
    existing_files = find_csv_files()

    validated_files = np.array([])

    missing_files = list(set(expected_files) - set(existing_files))
    extra_files = list(set(existing_files) - set(expected_files))
    common_files = list(set(existing_files) & set(expected_files))

    if not existing_files:
        print("Error: No csv files found. Please make sure this script shares",
              "the same file location. Exiting...")
        sys.exit()

    # Assumes that if EXPECTED_FILES are present, they are automatically validated
    if common_files:
        validated_files = np.append(validated_files, common_files)

    # Asks user if they want to proceed with missing EXPECTED_FILES
    if missing_files:
        validated_files = validate_missing_files(validated_files, missing_files)

    # Asks user if they want to include any other files in directory
    if extra_files:
        validated_files = validate_extra_files(validated_files, extra_files)

    # Takes care of edge case where can't find any expected files but found
    # other csv files none of which the user wants to use
    if len(validated_files) == 0:
        print("No files have been validated by user. Exiting...")
        sys.exit()

    return validated_files


def find_csv_files():
    """
    Finds any csv files in the directory of this script

    Returns
    -------
    list
        Existing files in directory

    """
    current_directory = os.path.dirname(os.path.realpath(__file__))
    return [file for file in os.listdir(current_directory) if file.endswith('.csv')]


def validate_missing_files(validated_files, missing_files):
    """
    Requests user input for missing files

    Parameters
    ----------
    validated_files : array
        expected files that are present in directory and additionally found
        files that have been approved by user
    missing_files : array
        all missing files from expected files

    Returns
    -------
    validated_files : array
        expected files that are present in directory and additionally found
        files that have been approved by user

    """
    user_input = str(input(f"Error: Expected {missing_files} but unable to find"
                           " them. Do you want to proceed without? (y/n): "))
    valid_input = False
    while not valid_input:
        if user_input.lower() == 'n' or user_input.lower() == 'no':
            print("Exiting...")
            sys.exit()
        elif user_input.lower() == 'y' or user_input.lower() == 'yes':
            valid_input = True
        else:
            user_input = str(input("Invalid input: Please type either yes or no: "))
    return validated_files


def validate_extra_files(validated_files, extra_files):
    """
    Requests user input for every single additional file that is not an
    expected file

    Parameters
    ----------
    validated_files : array
        expected files that are present in directory and additionally found
        files that have been approved by user
    extra_files : array
        all additonal files that are not expected files

    Returns
    -------
    validated_files : array
        expected files that are present in directory and additionally found
        files that have been approved by user

    """
    print("Warning: Unexpected additional files found in the directory:",
              f" {', '.join(extra_files)}")

    for file in extra_files:
        valid_input = False
        user_input = input(f"Do you want to include file {file}? (y/n): ")
        while not valid_input:
            if user_input.lower() == 'n' or user_input.lower() == 'no':
                valid_input = True
            elif user_input.lower() == 'y' or user_input.lower() == 'yes':
                validated_files = np.append(validated_files, file)
                valid_input = True
            else:
                user_input = str(input("Invalid input: Please type either yes or no: "))

    return validated_files


def read_and_check_files(validated_files):
    """
    Uses user validated files or expected files to read data. Removes text and
    nans, uncertainties that are 0 or negative, duplicates, and then sorts data
    
    Returns errors if files do not conform to [x, y, uncertainty in y], or
    if file cannot be read or accessed

    Parameters
    ----------
    validated_files : array
        File names in directory that have been approved by user for use

    Returns
    -------
    data : array like
        shape [x, y, uncertainty in y]

    """
    data = np.zeros([0,3])

    # Extract data from user confirmed files and replace non floats by nan
    for file_name in validated_files:
        try:
            input_file = np.genfromtxt(file_name, delimiter = ',',
                                       skip_header = 0)
            data = np.vstack((data, input_file))

            # Remove text and nans
            data = data[~np.isnan(data).any(axis=1)]

            # Remove zeros and negatives in uncertainty
            zero_index = np.where(data[:,2] <= 0)[0]
            data = np.delete(data, zero_index, 0)

            # Remove duplicates
            _, unique_indices = np.unique(data[:, 0], return_index=True)
            data = data[unique_indices]


            # Sort data from lowest energy to highest energy
            data = data[np.lexsort((data[:, 0],))]

        # Error messages
        except ValueError:
            print(f"Error: Data in {file_name} does not conform to required",
                  " style [x | y | uncertainty in y]. Exiting...")
            sys.exit()
        except FileNotFoundError:
            print("Error: The file {file_name} could not be accessed.",
                  " Exiting...")
            sys.exit()
        except IndexError:
            print(f"Error: The file {file_name} could not be read. Exiting...")
            sys.exit()
    return data


def gaussian(data, mean, stddev):
    """
    Defines a gaussian curve

    Parameters
    ----------
    data : array like
    mean : float
    stddev : float
        standard deviation

    Returns
    -------
    array
        yvals of gaussian for any xvals

    """
    return np.exp(-((data - mean) / stddev)**2 / 2)


def fit_gaussian(data):
    """
    Makes an initial general guess for boson mass and boson width by fitting a
    gaussian to the data and extracting the peak xval for boson mass and fwhm
    for the boson width. 
    For the gaussian fit function, p0 is the average of the data and an arbitrary 1.

    Parameters
    ----------
    data : array like
        shape [x, y, uncertainty in y]

    Returns
    -------
    mass_estimate: float
    width_estinate: float

    """
    initial_mass_initial_guess = np.mean(data[:,0]) # Mean of data as p0 parameter
    try:
        mass_estimate, stddev = sc.curve_fit(gaussian, data[:, 0], data[:, 1],
                                        sigma=data[:, 2], p0=[
                                            initial_mass_initial_guess, 1],
                                        absolute_sigma=True)[0]

        width_estimate = 2 * np.sqrt(2 * np.log(2)) * stddev
        print(f"\nDetermined an estimate for boson mass = {np.round(mass_estimate, 1)}",
              f"GeVc^-2 and boson width = {np.round(width_estimate, 1)} GeV")
        return mass_estimate, width_estimate
    except RuntimeError:
        print("\nAn estimate for the boson mass and width could not be found.",
              f"Proceeding with the standard boson mass ∼ {BOSON_MASS_EST} GeVc^-2 and",
              f"boson width ∼ {BOSON_WIDTH_EST} GeV")
        return BOSON_MASS_EST, BOSON_WIDTH_EST


def cross_section(energy, z_boson_mass, z_boson_width):
    """
    Calculates the cross section for each value of the Centre of Mass Energy
    
    sigma = (12pi * E^2 * Gamma_ee^2) / (m_Z^2 (E^2 - m_Z^2) + m_Z^2 Gamma_Z^2)
    where sigma: cross_section, E: energy, Gamma_ee: PARTIAL_WIDTH, 
                m_Z: boson_mass, Gamma_Z: boson_width

    Parameters
    ----------
    energy : numpy.ndarray
        first row of data array: data[:,0]
    z_boson_mass : float
        boson_mass
    z_boson_width : float
        boson_width

    Returns
    -------
    cross_section: function

    """
    top_fraction = (12 * np.pi / z_boson_mass**2 * energy**2 *
                    PARTIAL_WIDTH**2 * NATURAL_UNIT_CONVERSION * 1e9)
    bottom_fraction = ((energy**2 - z_boson_mass**2)**2 +
                        z_boson_mass**2 * z_boson_width**2)
    return top_fraction / bottom_fraction


def fit_function(data, mass_estimate, width_estimate):
    """
    Uses data input to return a best fit function value for each data point

    Parameters
    ----------
    data : np.array
        shape [x, y, uncertainty in y].

    Returns
    -------
    fit : np.array
        theoretical_values for each x value
    fit_parameters : np.array
        shape [estimate of boson mass, estimate of boson width]
    covariance_matrix : np.array # Not used
        2 x 2 matrix

    """
    fit_parameters, _ = sc.curve_fit(cross_section, data[:,0],
                                                     data[:,1],
                                                     p0=[mass_estimate,
                                                         width_estimate],
                                                     sigma=data[:,2],
                                                     absolute_sigma=True)
    fit = cross_section(data[:,0], fit_parameters[0], fit_parameters[1])
    return fit, fit_parameters


def plot_data(data, theoretical_values, outlier_matrix,
              minimum_reduced_chi_squared, fit_parameters):
    """
    Creates and saves a scatter plot of the data and the fit function

    Parameters
    ----------
    data : np.array
        shape [x, y, uncertainty in y].
    theoretical_values : np.array
        shape [y]

    Returns
    -------
    None.

    """
    # Plots figure of data
    plt.figure(1)
    plt.errorbar(data[:,0], data[:,1], yerr = data[:,2],
                 elinewidth = 1 ,fmt = 'x', label = "Data values",
                 color = '#6c5b7b', ecolor = '#355c7d')
    plt.plot(data[:,0], theoretical_values, label = "Fit function",
             color = '#355c7d')
    plt.errorbar(outlier_matrix[:, 0], outlier_matrix[:, 1],
                 yerr = outlier_matrix[:, 2], elinewidth = 1, fmt = 'x',
                 color = '#f67280', ecolor = '#f8b195', label = "Outliers")
    # axes attributes
    plt.xlabel("Centre of Mass Energy E (GeV)")
    plt.ylabel("Cross Section $\\sigma$ (nb)")
    plt.text(0.95, 0.95,
             (f"$\\chi^2_R$ = {np.round(minimum_reduced_chi_squared[0][0], 3)}"
             f"\n $m_z$ = {fit_parameters[0]:.4g} \n"
             f" $\\Gamma_Z$ = {fit_parameters[1]:.4g}"),
             ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3',
                                             edgecolor='black', alpha = 0.1),
             transform=plt.gca().transAxes)
    plt.legend(loc = 'upper left', shadow = True)
    plt.title("Assigmnet 2: $Z_0$ boson", fontweight = 'bold')
    plt.grid(True, linestyle = (0, (3, 5, 1, 5)), linewidth = 0.5,
             color = 'grey')
    # Limit such that outliers are ignored
    plt.ylim(0, max(data[:, 1]) + 3 * max(data[:, 1]) / len(data[:, 1]))
    plt.savefig(NAME_OF_FIGURE_1, dpi = 300)
    print(f"Saved figure of σ over E as {NAME_OF_FIGURE_1}")
    plt.show()


def remove_outliers(data, mass_estimate, width_estimate):
    """
    Removes outliers and fits function
    Does an initial fit and checks if there are any data points outside 3 stds
    if there are, it takes the outlier furthest from fit and removes it from 
    data_strip. Repeats until all data points are within 3 stds from fit

    Parameters
    ----------
    data : array like

    Returns
    -------
    data_strip : data stripped of outliers
        same shape as data but removed data points along axis 1
    theoretical_values : array like
        data points given the best fit parameters calculated by theoretical
        formula
    fit_parameters : array
        best fit parameters = [boson_mass, boson_width]
    outlier_matrix: array like
        all removed data points

    """
    # Initial first fit
    data_strip = np.array(data)
    theoretical_values, fit_parameters = fit_function(
        data_strip, mass_estimate, width_estimate)
    difference_matrix = np.array(abs(theoretical_values - data_strip[:, 1]))
    boolian_matrix = difference_matrix > 3 * data_strip[:, 2]
    outlier_matrix = np.empty((0,3))

    # Runs loop until boolian_matrix apprehends no data beyond three stdds
    while np.any(boolian_matrix):
        # Determine highest outlier and its index
        outliers = data_strip[:,1][boolian_matrix]

        max_outlier_index = np.argmax(outliers)
        max_outlier_data_index = np.where(boolian_matrix)[0][max_outlier_index]

        # Delete outliers from data
        outlier_matrix = np.vstack((outlier_matrix, data_strip[
            max_outlier_data_index]))
        data_strip = np.delete(data_strip, max_outlier_data_index, axis = 0)

        # Reevaluate boolian matrix
        theoretical_values, fit_parameters = fit_function(
            data_strip, mass_estimate, width_estimate)
        difference_matrix = np.array(abs(
            theoretical_values - data_strip[:, 1]))
        boolian_matrix = difference_matrix > 3 * data_strip[:, 2]
    print("Removed outliers:\n", outlier_matrix, "\n")
    return data_strip, theoretical_values, fit_parameters, outlier_matrix


def calculate_chi_squared(data, theoretical_values):
    """
    Uses data input and theoretical predicted values to calculate chi squared
    Needs to be reshape to be compatible with meshes of chi_squared plot

    Parameters
    ----------
    data : np.array 
        2D array of shape [x, y, uncertainty in y]
    theoretical_values : np.array
        calculated using fit parameters

    Returns
    -------
    chi_squared: float or array like
    

    """
    return np.sum(((data[:,1].reshape(1, 1, len(data)) - theoretical_values) /
                   data[:,2].reshape(1, 1, len(data)))**2, axis = 2)


def calculate_reduced_chi_squared(chi_squared, data):
    """
    Reduced Chi squared using standard formula
    chi^2 / N - n
    where N is the number of data points and n is the number of parameters

    Parameters
    ----------
    chi_squared : float
    data : array like

    Returns
    -------
    reduced_chi_squared: float

    """
    return chi_squared / (len(data) - NUMBER_OF_PARAMETERS)


def calculate_fit_estimates(data_strip, fit_parameters,
                            minimum_reduced_chi_squared):
    """
    Calculates an initial guess of fit parameters assuming no covariance until
    2.5 standard deviations than minimum_chi_squared. This should allow the 
    ellipse to always lay comfortably in the plot. 

    Parameters
    ----------
    data_strip : array like
    fit_parameters : array
        array of best fit parameters [boson_mass, boson_width]

    minimum_reduced_chi_squared : float
        

    Returns
    -------
    mass_estimate : float
        estimated uncertainty in boson_mass
    width_estimate : float
        estimated uncertainty in boson_width.

    """
    difference_mass = 0
    difference_width = 0
    mass_estimate = 0
    width_estimate = 0

    # Estimate for boson_mass uncertainty
    while difference_mass < 2.5 / (len(data_strip) - NUMBER_OF_PARAMETERS):
        mass = abs(fit_parameters[0] - mass_estimate)
        fit = cross_section(data_strip[:, 0], mass, fit_parameters[1])

        chi_squared1 = calculate_chi_squared(data_strip, fit)
        reduced_chi_squared1 = calculate_reduced_chi_squared(chi_squared1,
                                                             data_strip)

        difference_mass = abs(minimum_reduced_chi_squared - reduced_chi_squared1)
        mass_estimate += 0.0001

    # Estimate for boson_width uncertainty
    while difference_width < 2.5 / (len(data_strip) - NUMBER_OF_PARAMETERS):
        width = abs(fit_parameters[1] - width_estimate)
        fit = cross_section(data_strip[:, 0], fit_parameters[0], width)

        chi_squared2 = calculate_chi_squared(data_strip, fit)

        reduced_chi_squared2 = calculate_reduced_chi_squared(chi_squared2,
                                                             data_strip)

        difference_width = abs(minimum_reduced_chi_squared - reduced_chi_squared2)

        width_estimate += 0.001

    return mass_estimate, width_estimate


def generate_reduced_chi_squared_mesh(data_strip, fit_parameters,
                                      fit_parameters_estimates):
    """
    Calculates 2D array of reduced chi squared values given a range in fit
    parameters boson mass and boson width. Range is determined by constant

    Parameters
    ----------
    data_strip : array like
        data without outliers
    fit_parameters : array
        array of best fit parameters [boson_mass, boson_width]
    fit_parameters_estimates : array
        array of best fit parameters [mass_estimate, width_estimate]
    Returns
    -------
    mass_mesh : arry like
        2D array of boson_mass
    width_mesh : array like
        2D array of boson_width
    chi_squared_mesh : 2D array
        chi^2 values for any combination of mass and width

    """

    mass, width = np.meshgrid(
        np.linspace(fit_parameters[0] - fit_parameters_estimates[0],
                    fit_parameters[0] + fit_parameters_estimates[0], 100),
        np.linspace(fit_parameters[1] - fit_parameters_estimates[1],
                    fit_parameters[1] + fit_parameters_estimates[1], 100))

    mass = mass[:, :, np.newaxis]
    width = width[:, :, np.newaxis]

    # Needs to be reshaped to be broadcasted with parameter mesh values
    mass_mesh = mass.reshape(len(mass), -1)
    width_mesh = width.reshape(len(width), -1)

    fit = cross_section(data_strip[:, 0], mass, width)

    chi_squared_mesh = calculate_chi_squared(data_strip, fit)

    return mass_mesh, width_mesh, chi_squared_mesh


def plot_chi_squared(data, mass_mesh, width_mesh, chi_squared_mesh,
                     fit_parameters, minimum_reduced_chi_squared):
    """
    Produce plot of chi squared, PLot local minimum, the uncertainties, and
    represents relevant information on the side

    Parameters
    ----------
    data : array like
        data without outliers
    mass_mesh : array like
        2D array of boson_mass
    width_mesh : array like
        2D array of boson_wdith
    chi_squared_mesh : array like
        2D array of chi^2 values for different values of mass and width
    fit_parameters : array
        array of best fit parameters [boson_mass, boson_width].
    minimum_reduced_chi_squared : float

    Returns
    -------
    contour_lines : QuadContourSet
        Stores the contour line in the chi squared plot corresponding to chi^2
        +1

    """
    plt.figure(1, figsize=(8, 4), dpi=400)
    plt.title('$\\chi^2$ over $m_z$ and $\\Gamma_z$', size=19)

    # Plot all figures
    pcm = plt.pcolormesh(mass_mesh, width_mesh, chi_squared_mesh, cmap='magma')
    plt.colorbar(pcm)
    chi_squared_contour = (minimum_reduced_chi_squared[0][0] *
        (len(data) - NUMBER_OF_PARAMETERS) + 1)
    contour_lines = plt.contour(mass_mesh, width_mesh, chi_squared_mesh,
                                levels = [chi_squared_contour],
                                colors = 'white', linewidths=1)

    plt.scatter([fit_parameters[0]], [fit_parameters[1]], color='white',
                marker='x', label='Best fit parameter')

    contour_coordinates = contour_lines.collections[0].get_paths()[0].vertices
    plt.vlines(np.min(contour_coordinates[:, 0]),
               np.min(contour_coordinates[:, 1]),
               np.max(contour_coordinates[:, 1]),
               colors = 'black', linewidth = 2, linestyles = 'dashed')
    plt.hlines(np.max(contour_coordinates[:, 1]),
               np.min(contour_coordinates[:, 0]),
               np.max(contour_coordinates[:, 0]),
               colors = 'grey', linewidth = 2, linestyles = 'dashed')

    # Label of contour line
    manual_labels = {contour_lines.levels[0]: '$\\chi^2_{{min}} + 1$'}
    manual_locations = [(fit_parameters[0], fit_parameters[1])]
    plt.clabel(contour_lines, inline=True, manual = manual_locations,
               fontsize=14, fmt=manual_labels)

    # Define and plot legend
    scatter_proxy = Line2D([0], [0], marker='x', color='white',
                           linestyle = 'None',
                           label= f'''Minimum $\\chi^2$:\n{round(
                               minimum_reduced_chi_squared[0][0] * 
                               (len(data)-NUMBER_OF_PARAMETERS), 3)}''')
    contour_proxy = Line2D([0], [0], color='white',
                           label='$\\chi^2_{{min}}$ + 1')
    vlines_proxy = Line2D([0], [0], color='black', linestyle = 'dashed',
                          label= f'''$\\sigma_{{m_z}}$: {np.round(
                              calculate_parameter_uncertainties(
                                  contour_lines)[0],2)}''')
    hlines_proxy = Line2D([0], [0], color='grey', linestyle = 'dashed',
                          label= f''''$\\sigma_{{\\Gamma_z}}$: {np.round(
                              calculate_parameter_uncertainties(
                                  contour_lines)[1],3)}''')
    plt.legend(handles=[scatter_proxy, contour_proxy, vlines_proxy,
                        hlines_proxy], loc='lower right')

    plt.xlabel("boson mass $m_z$")
    plt.ylabel("boson width $\\Gamma_z$")
    plt.savefig(f"{NAME_OF_FIGURE_2}", dpi = 300)
    print(f"Saved figure of χ² over m₂ and Γ₂ as {NAME_OF_FIGURE_2}\n")
    plt.show()
    return contour_lines


def calculate_parameter_uncertainties(contour_lines):
    """
    Find leftmost and highest point of the uncertainty ellipse which correspond
    to double the uncertainty in each axis

    Parameters
    ----------
    contour_lines : QuadContourSet
        Stores the contour line in the chi squared plot corresponding to chi^2
        +1.

    Returns
    -------
    uncertainty_boson_mass : float
        uncertainty in boson_mass
    uncertainty_boson_width : float
        uncertainty in boson_width

    """
    # Extract most extreme points
    contour_coordinates = contour_lines.collections[0].get_paths()[0].vertices
    min_mass = np.min(contour_coordinates[:, 0])
    max_mass = np.max(contour_coordinates[:, 0])
    min_width = np.min(contour_coordinates[:, 1])
    max_width = np.max(contour_coordinates[:, 1])

    # Uncertainty
    uncertainty_boson_mass = (max_mass - min_mass) / 2
    uncertainty_boson_width = (max_width - min_width) / 2
    return uncertainty_boson_mass, uncertainty_boson_width


def calculate_boson_lifetime(fit_parameters, boson_width_uncertainty):
    """
    Converts units and calcualtes lifetime of boson. Assumes that all units are
    as given by the assigment

    Parameters
    ----------
    fit_parameters : array
        array of best fit parameters [boson_mass, boson_width]
    boson_width_uncertainty : float
        uncertainty in boson_width.

    Returns
    -------
    boson_lifetime : float
        uncertainty in boson_width.
    boson_lifetime_uncertainty : TYPE
        DESCRIPTION.

    """
    boson_lifetime = const.hbar / (const.e * fit_parameters[1] * 10**9)
    boson_lifetime_uncertainty = boson_lifetime * (boson_width_uncertainty /
                                                   fit_parameters[1])
    return boson_lifetime, boson_lifetime_uncertainty


def print_format_output_bosons(fit_parameters, boson_mass_uncertainty,
                            boson_width_uncertainty):
    """
    Prints and formats the boson_mass and boson_width and their uncertainties

    Parameters
    ----------
    fit_parameters : array
        array of best fit parameters [boson_mass, boson_width]
    boson_mass_uncertainty : float
        uncertainty in boson_mass
    boson_width_uncertainty : float
        uncertainty in boson_width

    Returns
    -------
    None.

    """
    print("-" * 75)

    # Format boson mass
    boson_mass_str = f"{fit_parameters[0]:.4g}"
    number = count_decimal_places(boson_mass_str)
    print("Boson mass: ", boson_mass_str, "+/-",
          f"{boson_mass_uncertainty:.{number}f} Gevc^-2" )

    # Format boson width
    boson_width_str = f"{fit_parameters[1]:.4g}"
    number = count_decimal_places(boson_width_str)
    print("Boson width: ", boson_width_str, "+/-",
          f"{boson_width_uncertainty:.{number}f} GeV")

def print_format_output_lifetime(boson_lifetime, boson_lifetime_uncertainty,
                              minimum_reduced_chi_squared):
    """
    Prints and formats the lifetime of the boson and the uncertainty

    Parameters
    ----------
    boson_lifetime : float
    boson_lifetime_uncertainty : float
    minimum_reduced_chi_squared : array
        (1, 1).

    Returns
    -------
    None.

    """
    # Automatically formats lifetime of from a^x +/- b^(x+c) such that (a +/- b * 10^-c) * 10^x
    boson_lifetime_str, exponent = f"{boson_lifetime:.3g}".split('e')
    number = count_decimal_places(boson_lifetime_str)
    boson_lifetime_uncertainty_str, uncertainty_exponent = (
        f"{boson_lifetime_uncertainty:.{number - 1}g}".split('e'))

    exponent = int(exponent)
    uncertainty_exponent = int(uncertainty_exponent)
    boson_lifetime_uncertainty_str = float(boson_lifetime_uncertainty_str)

    print(f"Lifetime: ({boson_lifetime_str} +/-",
          f""" {boson_lifetime_uncertainty_str * 10**(-abs(exponent - 
            uncertainty_exponent))}) * 10^{exponent} s""")

    print("\nReduced Chi-squared:",
          f" {round(minimum_reduced_chi_squared[0][0], 4)}")


def count_decimal_places(parameters):
    """
    Counts decimal places of parameters
    Relevant if a parameter is rounded to sig. dig. and uncertainty needs to be
    given to the same decimal place

    Parameters
    ----------
    parameters : str

    Returns
    -------
    float
        Number of decimal places

    """
    if '.' in parameters:
        return len(parameters.split('.')[1])
    return 0


### MAIN CODE EXECUTION ###
def main():
    """
    Calls each function and outputs the local variable

    Returns
    -------
    None.

    """
    print("-" * 76,"\nThe aim of this project is to deduce the mass, width,",
          "and lifetime of a Z_0 \nboson using two data files of centre of",
          "mass energy, cross section and its \nuncertainty.\n", "-" * 75, 
          "\n")

    # Set initial time
    time_start = time.time()

    # Read and validate data
    validated_files = validate_csv_files(EXPECTED_FILES)
    data = read_and_check_files(validated_files)

    # Initial guess of mass and width using gaussian
    mass_estimate, width_estimate = fit_gaussian(data)

    # Data processing
    data_strip, theoretical_values, fit_parameters, outlier_matrix = (
        remove_outliers(data, mass_estimate, width_estimate))

    # Calculate chi squared and reduced chi squared
    minimum_chi_squared = calculate_chi_squared(data_strip, theoretical_values)
    minimum_reduced_chi_squared  = calculate_reduced_chi_squared(
        minimum_chi_squared, data_strip)

    # Plot parameter data
    plot_data(data_strip, theoretical_values, outlier_matrix,
              minimum_reduced_chi_squared, fit_parameters)

    # Calculate rough uncertainty guess
    fit_parameters_estimates = calculate_fit_estimates(data_strip, fit_parameters,
                                  minimum_reduced_chi_squared)

    # calculate uncertainty
    mass_mesh, width_mesh, chi_squared_mesh = (
        generate_reduced_chi_squared_mesh(data_strip, fit_parameters,
                                          fit_parameters_estimates))
    contour_lines = plot_chi_squared(data_strip, mass_mesh, width_mesh,
                                     chi_squared_mesh, fit_parameters,
                                     minimum_reduced_chi_squared)
    boson_mass_uncertainty, boson_width_uncertainty = (
        calculate_parameter_uncertainties(contour_lines))

    # Calculate lifetime
    boson_lifetime, boson_lifetime_uncertainty = calculate_boson_lifetime(
        fit_parameters, boson_width_uncertainty)

    # Print
    print_format_output_bosons(fit_parameters, boson_mass_uncertainty,
                               boson_width_uncertainty)
    print_format_output_lifetime(boson_lifetime, boson_lifetime_uncertainty,
                                 minimum_reduced_chi_squared)

    # Calculate and display runtime (not CPU time and hence only an estimate)
    time_end = time.time() - time_start
    print(f"\nRuntime: {time_end:.2g} s")


if __name__ == "__main__":
    main()
