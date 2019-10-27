import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_uncertainties(num_tasks, all_ibp_uncerts, all_vcl_h5_uncerts, all_vcl_h10_uncerts,
                       all_vcl_h50_uncerts, tag):

    num_results = 4
    # Uncertainties
    all_ibp_uncerts_norm = np.mean(all_ibp_uncerts, axis=0) / np.tile(np.max(np.mean(all_ibp_uncerts, axis=0), axis=1),
                                                                      (num_tasks, 1)).T
    all_vcl_h5_uncerts_norm = np.mean(all_vcl_h5_uncerts, axis=0) / np.tile(
        np.max(np.mean(all_vcl_h5_uncerts, axis=0), axis=1),
        (num_tasks, 1)).T
    all_vcl_h10_uncerts_norm = np.mean(all_vcl_h10_uncerts, axis=0) / np.tile(
        np.max(np.mean(all_vcl_h10_uncerts, axis=0), axis=1),
        (num_tasks, 1)).T
    all_vcl_h50_uncerts_norm = np.mean(all_vcl_h50_uncerts, axis=0) / np.tile(
        np.max(np.mean(all_vcl_h50_uncerts, axis=0), axis=1),
        (num_tasks, 1)).T

    lw = 2
    grid_color = '0.1'
    grid_lw = 0.2
    title_size = 16
    label_size = 22
    tick_size = 14
    legend_size = 16

    fig, ax = plt.subplots(1, 4, figsize=(5, 3))
    for i in range(num_tasks):
        ax[0].plot(np.arange(len(all_ibp_uncerts_norm[i, :])) + 1, all_ibp_uncerts_norm[i, :], label='Task {}'.format(i),
                marker='o', linewidth=lw)
        ax[0].set_xticks(range(1, len(all_ibp_uncerts_norm[i, :])))

    for i in range(num_tasks):
        ax[1].plot(np.arange(len(all_vcl_h5_uncerts[i, :])) + 1, all_vcl_h5_uncerts[i, :], label='Task {}'.format(i),
                marker='o', linewidth=lw)
        ax[1].set_xticks(range(1, len(all_vcl_h5_uncerts[i, :])))

    for i in range(num_tasks):
        ax[2].plot(np.arange(len(all_vcl_h10_uncerts[i, :])) + 1, all_vcl_h10_uncerts[i, :], label='Task {}'.format(i),
                marker='o', linewidth=lw)
        ax[2].set_xticks(range(1, len(all_vcl_h10_uncerts[i, :])))

    for i in range(num_tasks):
        ax[3].plot(np.arange(len(all_vcl_h50_uncerts[i, :])) + 1, all_vcl_h50_uncerts[i, :], label='Task {}'.format(i),
                marker='o', linewidth=lw)
        ax[3].set_xticks(range(1, len(all_vcl_h50_uncerts[i, :])))

    for i in range(num_results):
        ax[i].set_xlabel('Tasks', fontsize=legend_size)
        ax[i].tick_params(labelsize=tick_size)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i].legend(fontsize=legend_size)
        if i == 0:
            ax[i].set_ylabel('Uncertainty', fontsize=legend_size)
    plt.savefig("plots/uncerts_{}.pdf".format(tag), bbox_inches='tight')
    fig.show()