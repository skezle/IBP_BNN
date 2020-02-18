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

    fig, ax = plt.subplots(1, 4, figsize=(14, 3))
    for i in range(num_tasks):
        ax[0].plot(np.arange(len(all_ibp_uncerts_norm[i, :])) + 1, all_ibp_uncerts_norm[i, :],
                   label='Task {}'.format(i),
                   marker='o', linewidth=lw)
        ax[0].set_xticks(range(1, len(all_ibp_uncerts_norm[i, :]) + 1))
        ax[0].set_title("IBP + VCL")

    for i in range(num_tasks):
        ax[1].plot(np.arange(len(all_vcl_h5_uncerts_norm[i, :])) + 1, all_vcl_h5_uncerts_norm[i, :],
                   label='Task {}'.format(i),
                   marker='o', linewidth=lw)
        ax[1].set_xticks(range(1, len(all_vcl_h5_uncerts_norm[i, :]) + 1))
        ax[1].set_title("VCL h5")

    for i in range(num_tasks):
        ax[2].plot(np.arange(len(all_vcl_h10_uncerts_norm[i, :])) + 1, all_vcl_h10_uncerts_norm[i, :],
                   label='Task {}'.format(i),
                   marker='o', linewidth=lw)
        ax[2].set_xticks(range(1, len(all_vcl_h10_uncerts_norm[i, :]) + 1))
        ax[2].set_title("VCL h10")

    for i in range(num_tasks):
        ax[3].plot(np.arange(len(all_vcl_h50_uncerts_norm[i, :])) + 1, all_vcl_h50_uncerts_norm[i, :],
                   label='Task {}'.format(i),
                   marker='o', linewidth=lw)
        ax[3].set_xticks(range(1, len(all_vcl_h50_uncerts_norm[i, :]) + 1))
        ax[3].set_title("VCL h50")

    for i in range(num_results):
        ax[i].set_xlabel('Tasks', fontsize=legend_size)
        ax[i].tick_params(labelsize=tick_size)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if i == num_results - 1:
            ax[i].legend(fontsize=legend_size)
        if i == 0:
            ax[i].set_ylabel('Uncertainty', fontsize=legend_size)
    plt.savefig("plots/uncerts_{}.pdf".format(tag), bbox_inches='tight')
    fig.show()

def plot_Zs(num_tasks, num_layers, Zs, dataset, tag):
    # this is plotting the final run Zs
    lw = 2

    grid_color = '0.1'
    grid_lw = 0.2

    title_size = 16
    label_size = 16
    tick_size = 12
    legend_size = 12
    no_active_neurons = [np.mean(np.asarray(np.squeeze(Zs[i] > 0.1)).astype(int)) for i in range(len(Zs))]
    if num_layers == 1:
        fig, ax = plt.subplots(2, num_tasks, figsize=(16, 4))
        for i in range(num_tasks):
            imgplot = ax[0][i].imshow(np.squeeze(Zs[i])[:50, :], cmap=plt.cm.Greys, vmin=0, vmax=1)
            ax[0][i].set_xticks(np.arange(0.0, 100, step=50))
            ax[0][i].set_xlabel('$k$', fontsize=legend_size)
            ax[0][i].set_title('No active neurons: {:.3f}'.format(no_active_neurons[i]))
            ax[0][i].set_xticklabels([])
            ax[0][i].set_yticklabels([])
            ax[1][i].hist(np.sum(np.asarray(np.squeeze(Zs[i] > 0.1)).astype(int), axis=1), 8, alpha=0.7,
                          edgecolor='green',
                          linewidth=1.5)
            ax[1][i].set_yticklabels([])
            ax[1][i].set_xlabel("Task {}".format(i + 1))
            if i == 4:
                cbar_ax = fig.add_axes([0.92, 0.57, 0.01, 0.28])
                fig.colorbar(imgplot, cax=cbar_ax)
    elif num_layers == 2:
        fig, ax = plt.subplots(4, num_tasks, figsize=(16, 8))
        for i in range(num_tasks):
            imgplot1 = ax[0][i].imshow(np.squeeze(Zs[2*i])[:50, :], cmap=plt.cm.Greys, vmin=0, vmax=1)
            ax[0][i].set_xticks(np.arange(0.0, 100, step=50))
            ax[0][i].set_xlabel('$k$', fontsize=legend_size)
            ax[0][i].set_xticklabels([])
            ax[0][i].set_yticklabels([])
            ax[0][i].set_title('L1 No active neurons: {:.3f}'.format(no_active_neurons[2*i]))
            ax[1][i].hist(np.sum(np.asarray(np.squeeze(Zs[2*i] > 0.1)).astype(int), axis=1), 8, alpha=0.7,
                          edgecolor='green',
                          linewidth=1.5)
            ax[1][i].set_yticklabels([])
            imgplot2 = ax[2][i].imshow(np.squeeze(Zs[(2 * i) + 1])[:50, :], cmap=plt.cm.Greys, vmin=0, vmax=1)
            ax[2][i].set_xticks(np.arange(0.0, 100, step=50))
            ax[2][i].set_xlabel('$k$', fontsize=legend_size)
            ax[2][i].set_xticklabels([])
            ax[2][i].set_yticklabels([])
            ax[2][i].set_title('L2 No active neurons: {:.3f}'.format(no_active_neurons[(2 * i) + 1]))
            ax[3][i].hist(np.sum(np.asarray(np.squeeze(Zs[2*i + 1] > 0.1)).astype(int), axis=1), 8, alpha=0.7,
                          edgecolor='green',
                          linewidth=1.5)
            ax[3][i].set_yticklabels([])
            ax[3][i].set_xlabel("Task {}".format(i + 1))
            if i == 4:
                cbar_ax = fig.add_axes([1.0, 0.78, 0.01, 0.2]) # left, bottom, width, height
                fig.colorbar(imgplot1, cax=cbar_ax)
                fig.colorbar(imgplot2, cax=cbar_ax)
    else:
        raise ValueError
    fig.tight_layout()
    plt.savefig('plots/Zs_{0}_mnist_{1}.pdf'.format(dataset, tag), bbox_inches='tight')
    fig.show()
