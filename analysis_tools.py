#!/usr/bin/env python3

__author__ = "Matthew R. Carbone & Erica J. Sturm" 
__maintainer__ = "Matthew R. Carbone"
__email__ = "x94carbone@gmail.com" #MRC
__status__ = "Prototype" # ?????????????? In development as I need stuff???


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from mlnrg.utils.logger import log

import scipy.signal

#-------------------------------------------------------------------------------

def plot_loss(set_loss, training_loss, parameters, set_name):
    """Takes the loss values for the training and validation/test sets and 
    plots them alongside each other with appropriate title.

    ***Only for NN results!!***

    Parameters
    ----------
    set_loss : list
        The validation or test set loss values.
    training_loss : list
        The training set loss values. 
    parameters : dict[str]
        The parameters for titling the graph.
    """ 

    plt.clf();
   
    plt.plot(set_loss, color='k', label=set_name);
    plt.plot(training_loss, color='r', linestyle=":", label="Training Set");

    plt.legend(framealpha=1);
    plt.yscale('log');
    plt.title(f'{parameters["dataset"]}');
    plt.xlabel("Epochs", fontsize=16);
    plt.ylabel("Loss", fontsize=16);

    plt.show()

#-------------------------------------------------------------------------------

def set_up_df(result_set, unscaled_data, data_set):
    """Construct results dataframe by initializing all relative fields in the
    validation or test set. We must sort by index first to prevent artifact 
    generation. 
    Parameters
    ----------
    result_set : list
        The target (and already loaded) validation or test set.

    unscaled_data : mlnrg.loader.NRGData
        The original dataframe without parameter scaling. Required for
        the calculation of the SPE.
   
    data_set : string
        Either 'kondo' or 'anderson'
        Tells the subroutine how to transform the relevant information into
        a dataframe. 

    Returns
    -------
    An object with the fields named and a numpy array of the validation set's 
    unscaled energies (B, T, TK)
    """

    num_trials = len(result_set);
    idx = [result_set[ii].name for ii in range(num_trials)];

    # Erica needs to fix this awful tangle of code--another day. 
    # Right now it works, fix it later. 
    if(data_set == 'anderson'):
        unscaled_energies = np.array(
                            unscaled_data.raw.loc[idx][['B', 'T', 'TK']]
                            );
        raw_TK = np.asarray(unscaled_data.raw.iloc[idx,:]["TK"])

    if(data_set == 'kondo'):
        idx_names_array = np.array(
                          [result_set[ii].name for ii in range(num_trials)]
                          );
        unscaled_energies = np.asarray([np.array(unscaled_data.raw.loc[
                            unscaled_data.raw['idx'] ==   
                            idx_names_array[ii]][['B', 'T', 'TK']]) \
                            for ii in range(num_trials)
                            ]
                            )
        unscaled_energies = np.reshape(unscaled_energies, [num_trials, 3]);

        raw_TK = np.asarray([np.array(unscaled_data.raw.loc[
                            unscaled_data.raw['idx'] ==   
                            idx_names_array[ii]][['TK']]) \
                            for ii in range(num_trials)
                            ]
                            );
        raw_TK = np.reshape(raw_TK, num_trials);

    SPE_array = np.log10(np.max(np.abs(unscaled_energies), axis=1));

    # TK was not placed on a log10 scale during the load, so it is done in 
    # this dataframe intitialization.
    results = pd.DataFrame(
        {
        'idx': [result_set[ii].name for ii in range(num_trials)],
        'U':   [result_set[ii].feature[0] for ii in range(num_trials)],
        'Gamma':  [result_set[ii].feature[1] for ii in range(num_trials)],
        'epsd': [result_set[ii].feature[2] for ii in range(num_trials)],
        'B': [result_set[ii].feature[3] for ii in range(num_trials)],
        'T': [result_set[ii].feature[4] for ii in range(num_trials)],
        'err':  [result_set[ii].mae for ii in range(num_trials)],
        'TK': list(np.log10(raw_TK)),
        'SPE': list(SPE_array)
        }
    )

    return results, unscaled_energies

#-------------------------------------------------------------------------------

def plot_err_histogram(errors):
    """Takes the validation errors. Computes error stats and plots them. 
    Returns stats.
    Parameters
    ----------
    errors : list
        The error (usualy from the validation set, but could be test set?) 
    Returns 
    -------
    The values for stats: deciles, mean, median.
    """
    
    deciles = np.percentile(np.log10(errors), [10 * ii for ii in range(1,10)]);
    median = np.percentile(np.log10(errors), 50);
    mean = np.mean(np.log10(errors)); 

    log.info("Validation set log10(mean) error: %.04g" % mean);
    log.info("Validation set log10(median) error: %.04g" % median);
    log.info("Validation set deciles (on log10 scale):\n%s" % deciles);

    # Now plot them.
    plt.clf();
    fig, ax = plt.subplots(1,1, figsize=(10,6));
    pop, edges, patch = ax.hist(np.log10(errors), bins=60, color='k');
    ylimits = ax.get_ylim();

    for ii, d in enumerate(deciles):
        ax.plot((d,d), ylimits, 'r', label="Deciles" if ii == 0 else None);

    ax.plot((median, median), ylimits, color='b', \
           label=f"Median: {median:.03f}"
           );
    ax.plot((mean, mean), ylimits, color='orange', \
           label=f"Mean: {mean:.03f}"
           );
    ax.set_xlabel(r"$\log_{10}\mathrm{MAE}$", fontsize=16);
    ax.set_ylabel("Occurances", fontsize=16);
    ax.legend(fontsize=14, frameon=False);
    
    return [deciles, mean, median];

#-------------------------------------------------------------------------------

def plot_param_best_and_worst_dist(results_df, N=100):
    """Take the validation dataframe and an integer number of trials. Plots all paramters values for the X best and worst trials on a histogram for analysis.
    Parameters
    ----------
    results_df : dataframe
        Validation set dataframe with all relevant parameters
  
    N : int
        The desired number of trials to display for best and worst histograms.
        Defaults to 100 values if not specified.
    """ 

    L = len(results_df['idx']);
    variable = ['U', 'Gamma', 'epsd', 'B', 'T', 'TK', 'SPE'];
    labels = ['U', '$\Gamma$', '$\epsilon_d$', '$\log_{10}|B|$',
             '$\log_{10}$T', '$\log_{10}T_K$', '$\log_{10}$SPE'
             ];

    plt.clf() 
    fig, ax = plt.subplots(2,len(variable), sharey=True, sharex=False, 
                          figsize=(25,8)
                          ); 

    for ii in range(len(variable)): 
        ax[0, ii].hist(results_df[variable[ii]].iloc[(L-N):],
                      bins=30, color='red',
                      )
 
        ax[1, ii].hist(results_df[variable[ii]].iloc[:N],
                      bins=30, color='b',
                      label=f'Best {N}'
                      )

        if(ii == 0):
            ax[0,ii].legend([f"Worst {N}"], loc='upper right', 
                           fontsize=12, frameon=False
                           )
            ax[1,ii].legend([f"Best {N}"], loc='upper right', 
                           fontsize=12, frameon=False
                           )

        minVal = results_df[variable[ii]].min();
        maxVal = results_df[variable[ii]].max();
        ax[0,ii].set_xlim(minVal, maxVal);
        ax[1,ii].set_xlim(minVal, maxVal);
        ax[1,ii].set_xlabel(r"%s" % labels[ii], fontsize=15)

    plt.show()

#-------------------------------------------------------------------------------

def plot_best_and_worst_SPE_control(results_df, unscaled_energies, set_Name):
    
    N = int(0.1*len(results_df.index));
    L = len(results_df.index);
    
    BisMax_best = np.array([]); BisMax_worst = np.array([]);
    TisMax_best = np.array([]); TisMax_worst = np.array([]);
    TKisMax_best = np.array([]); TKisMax_worst = np.array([]);
    
    abs_energies_best = np.abs(unscaled_energies[:N]);
    abs_energies_worst = np.abs(unscaled_energies[(L-N):]);
    
    for ii in range(len(abs_energies_best)):
        if np.argmax(abs_energies_best[ii]) == 0:
            BisMax_best = np.append(BisMax_best, 
                                   np.log10(np.amax(abs_energies_best[ii]))
                                   );
        if np.argmax(abs_energies_best[ii]) == 1:
            TisMax_best = np.append(TisMax_best,
                                   np.log10(np.amax(abs_energies_best[ii]))
                                   );
        if np.argmax(abs_energies_best[ii]) == 2:
            TKisMax_best = np.append(TKisMax_best, 
                                    np.log10(np.amax(abs_energies_best[ii]))
                                    );
            
    for jj in range(len(abs_energies_worst)):
        if np.argmax(abs_energies_worst[jj]) == 0:
            BisMax_worst = np.append(BisMax_worst, 
                                    np.log10(np.amax(abs_energies_worst[jj]))
                                    );
        if np.argmax(abs_energies_worst[jj]) == 1:
            TisMax_worst = np.append(TisMax_worst, 
                                    np.log10(np.amax(abs_energies_worst[jj]))
                                    );
        if np.argmax(abs_energies_worst[jj]) == 2:
            TKisMax_worst = np.append(TKisMax_worst, 
                                     np.log10(np.amax(abs_energies_worst[jj]))
                                     );  
            
    plt.clf();
    labels = ["$|B|$", "$T$", "$T_K$"];
    colors = ["blueviolet", "firebrick", "pink"];

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5), 
                                  sharey=True, sharex=False
                                  );
    ax1.hist([BisMax_best, TisMax_best, TKisMax_best], 
            bins=30, stacked=True, color=colors, edgecolor='k'
            );
    ax1.set_title("%d best %s trials" % (N, set_Name));
    ax1.set_ylabel("Number of Occurances", fontsize=12);
    ax1.legend(reversed(ax1.legend(labels, loc="upper left").legendHandles), 
              reversed(labels)
              );
    
    ax2.hist([BisMax_worst, TisMax_worst, TKisMax_worst], 
            bins=30, stacked=True, color=colors, edgecolor='k'
            );
    ax2.set_title("%d worst %s trials" % (N, set_Name));
    
    fig.text(0.5,0, r"log$_{10}(SPE)$ Value", ha='center', fontsize=12);
    plt.show()

#-------------------------------------------------------------------------------

def plot_percentile(valid_set, unscaled_data, percentile):
    """
    The user specifies a desired percentile between 0 and 100 inclusive. The
    first trial will be closest to that percentile and the remaining 9 trials
    displayed will be the next best ones after that. 
    Special percentile value 0 will call the absolute worst 10 trials.
    Special percentile value 100 will call the absolute best 10 trials. 
    *** Setting percentile=99 will provide the trial at the 99th percentile 
    which are not the best trials! The best are >99th percentile! ***
    Parameters
    ----------
    valid_set : list
        The original validation set list loaded straight from the pkl file. 
    unscaled_data : mlnrg.loader.NRGData
        The original dataframe without parameter scaling. Required for
        printing the displayed trials' physical parameters & ID.
    percentile : float \in [0, 100] (ints work too)
        The desired percentile to be shown. 
    """

    if ((percentile < 0) or (percentile > 100)):
        crit = "Invalid percentile input. Use 0 <= percentile <= 100.";
        log.error(crit);
        raise RuntimeError(crit);

    if (percentile == 0): 
    # Special case, worst 10 in BACKWARDS order, so last image is worst.
        title = "10 worst trials"
        addend = -10;
        prediction = [valid_set[ii].pred for ii in range(-10,0)];
        ground_truth = [valid_set[ii].target for ii in range(-10,0)];
        mae = [np.abs(np.array(prediction[ii]) - \
              np.array(ground_truth[ii])).mean() \
              for ii in range(10)
              ];

    elif (percentile == 100): 
    # Special case, best 10, starting with best as first image. 
        title = "10 best trials"
        addend = 0; 
        prediction = [valid_set[ii].pred for ii in range(10)];
        ground_truth = [valid_set[ii].target for ii in range(10)];
        mae = [np.abs(np.array(prediction[ii]) - \
              np.array(ground_truth[ii])).mean() \
              for ii in range(10)
              ];

    else: 
    # All other cases, always displayed best to worst for given percentile.
    # Because the results are stored best --> worst, to get the Nth 
    # percentile, one must take 100-N for the addend variable. 
        title = "10 trials from %.2g percentile" % percentile;
        addend = int(len(valid_set) * ((100-percentile)/100)); 
        prediction = [valid_set[ii + addend].pred for ii in range(-10,0)];
        ground_truth = [valid_set[ii + addend].target for ii in range(-10,0)];
        mae = [np.abs(np.array(prediction[ii]) - \
              np.array(ground_truth[ii])).mean() \
              for ii in range(10)
              ];
  
    index = [valid_set[ii + addend].name for ii in range(10)];
    U_val = [unscaled_data.raw[unscaled_data.raw['idx'] == \
            valid_set[ii + addend].name]['U'] \
            for ii in range(10)
            ]; 
    G_val = [unscaled_data.raw[unscaled_data.raw['idx'] == \
            valid_set[ii + addend].name]['Gamma'] \
            for ii in range(10)
            ]; 
    e_val = [unscaled_data.raw[unscaled_data.raw['idx'] == \
            valid_set[ii + addend].name]['eps'] \
            for ii in range(10)
            ]; 
    B_val = [unscaled_data.raw[unscaled_data.raw['idx'] == \
            valid_set[ii + addend].name]['B'] \
            for ii in range(10)
            ]; 
    T_val = [unscaled_data.raw[unscaled_data.raw['idx'] == \
            valid_set[ii + addend].name]['T'] \
            for ii in range(10)
            ]; 
    TK_val = [unscaled_data.raw[unscaled_data.raw['idx'] == \
            valid_set[ii + addend].name]['TK'] \
            for ii in range(10)
            ]; 

    plt.clf()
    fig, axs= plt.subplots(2,5, sharey=True, sharex=False, figsize=(16,8));
    bbox = dict(facecolor='white', edgecolor='white', boxstyle='round', 
               alpha=0.9
               );
    
    for ii in range(len(prediction)):
        axs[ii//5, ii%5].plot(ground_truth[ii], color='k', linewidth=2);
        axs[ii//5, ii%5].plot(prediction[ii], color='red', linewidth=2,
                             linestyle='--'
                             );
        axs[ ii//5, ii%5].text(
            0.05, 0.53,
            ' #%i \n MAE = %.02g \n $U$ = %.02g \n $\Gamma$ = %.02g \n'\
            ' $\epsilon$ = %.02g \n $B$ = %.02g \n $T$ = %.02g \n'\
            ' $T_K$ = %.02g'
            % (index[ii], mae[ii], U_val[ii], G_val[ii], e_val[ii],
            B_val[ii], T_val[ii], TK_val[ii]),
            fontsize=10,
            transform=axs[ii//5,ii%5].transAxes,
            bbox=bbox
            );
        if ii % 5 == 0:
            axs[ii//5, ii%5].set_ylabel("$A(\omega)*\pi\Gamma$", 
                                       fontsize=18
                                       );
        axs[ii//5, ii%5].set_xticks((np.arange(0,350, step=100)));

        if(percentile == 100):
            axs[0,0].set_title("Best Trial!", fontsize=12, color='b');
        if(percentile == 0):
            axs[1,4].set_title("Worst Trial!", fontsize=12, color='b');

    fig.text(0.5,0.05, "%s" % title, ha='center', fontsize=22);
    plt.show()

#-------------------------------------------------------------------------------
def plot_single_trial(unscaled_data, omegas,  selection):
    """
    Plots a user-selected trial from the FULL data set. No machine learning
    info here, so it's okay to access any single trail even if normally it 
    would be a test trial.
    Plots selected trial 3 different ways, always finds any local maxima
    and notes them. Also computes and displays SPE control parameter.
    Parameters
    ----------
    unscaled_data : mlnrg.loader.NRGData
        The original dataframe without parameter scaling. Required for
        printing the displayed trials' physical parameters & ID and 
        computing SPE.
    omegas : np.array
        The natural frequency values for the x-axis.
    selection : int
        The user-specified trial of interest; 1-indexed for the user,
        but converted back to 0-index for internal python usage here.
    """

    if ((selection < 1) or (selection > len(unscaled_data.raw.loc[:]['idx']))):
        crit = "Invalid selection input. Select positive integer up to %d" \
               % len(unscaled_data.raw.loc[:]['idx']);
        log.error(crit);
        raise RuntimeError(crit);

    features = unscaled_data.raw.loc[selection-1][:8];
    targets = unscaled_data.raw.loc[selection-1][8:];
    maxima, _ = scipy.signal.find_peaks(
                    targets, height=0.001, distance=1
                    );

    bbox = dict(facecolor='white', edgecolor='white', boxstyle='round', 
               alpha=0.9
               );

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5),
                               sharey=False, sharex=False
                               );

    # Left image; regular spectrum with natural (linear) omega spacing.
    ax1.plot(omegas, targets, color='k', linewidth=1);
    ax1.scatter(omegas[maxima], targets[maxima], color='b', marker='*');
    ax1.set_title("Spectral Function");
    ax1.set_ylim([0, 1.1]);
    ax1.set_yticks(np.arange(0, 1.1, step=0.1));
    ax1.set_ylabel("$A(\omega)*\pi\Gamma$", size=12);
    ax1.set_xlabel("$\omega$ (units of bandwidth)", fontsize=12)
    ax1.text(0.025,0.95, r"Max peak height = %.04g"
            % np.max(targets[maxima]), 
            fontsize=10, transform=ax1.transAxes, bbox=bbox
            )
    
    # Middle image; x-axis is plotted as integers, Y on symlog scale.
    intX = np.arange(len(targets));
    ax2.plot(intX, targets, color='k', linewidth=1);
    ax2.scatter(intX[maxima], targets[maxima], color='b', marker='*');
    ax2.set_title("Interger-spaced X axis with symlogY");
    ax2.set_ylim([0,1.1]);
    ax2.yaxis.set_visible(False);
    ax2.set_yscale('symlog');
    ax2.set_xlabel("Coordinate Index X (integers)", fontsize=12);
    
    # Right image; x-axis is (true omegas/SPE value).
    SPE = np.max(np.abs(features[['B', 'T', 'TK']]));
    if (SPE == float(np.abs(features[['B']]))):
        control = '|B|';
    if (SPE == float(np.abs(features[['T']]))):
        control = 'T';
    if (SPE == float(np.abs(features[['TK']]))):
        control = '$T_K$';
    rescaled_omegas = np.divide(omegas, float(SPE));
    
    ax3.plot(rescaled_omegas, targets, color='k', linewidth=1);
    ax3.scatter(rescaled_omegas[maxima], targets[maxima], color='b', 
               marker="*"
               );
    ax3.set_title("Rescaling of $\omega$ by SPE; Natural Y");
    ax3.set_xlabel("$\omega$/SPE");
    ax3.yaxis.set_visible(False);
    xLim = np.min([100, 0.8/SPE]);
    ax3.set_xlim([-1*xLim, xLim]);
    ax3.set_ylim([0, 1.1]);
    
    ax3.text(0.025,0.95, r"SPE: %s = %.04g" % (control, SPE), 
             fontsize=10,
             transform=ax3.transAxes, bbox=bbox
            );
    
    fig.text(0.5, 0.95, "Trial #%d\n$U$ = %.02g   $\Gamma$ = %.02g   "\
            "$\epsilon_d$ = %.02g   $B$ = %.02g   $T$ = %.02g   "\
            "$T_K$ = %0.02g"
             % (features[0], features[1], features[2],
                features[3], features[4], features[5], features[7]
               ),
             ha='center',
             fontsize=16
             );
    fig.text(0.5, -0.075, "Local maxima appear at $\omega = $ %s"\
            "\nOr coordinate X = %s out of 333." 
            % (omegas[maxima], intX[maxima]+1),
            fontsize=14, 
            color='b',
            ha='center'
            );

#-------------------------------------------------------------------------------
