from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import numpy as np
import os

def pom_frame(model, dataset, epoch, dir, is_preds=False, threshold=0.4):
    pom_embeds = model.predict_embedding(dataset)
    y_preds = model.predict(dataset)
    required_desc = list(dataset.tasks)
    type1 = {'floral': '#F3F1F7', 'subs': {'muguet': '#FAD7E6', 'lavender': '#8883BE', 'jasmin': '#BD81B7'}}
    type2 = {'meaty': '#F5EBE8', 'subs': {'savory': '#FBB360', 'beefy': '#7B382A', 'roasted': '#F7A69E'}}
    type3 = {'ethereal': '#F2F6EC', 'subs': {'cognac': '#BCE2D2', 'fermented': '#79944F', 'alcoholic': '#C2DA8F'}}
    
    # Assuming you have your features in the 'features' array
    pca = PCA(n_components=2, iterated_power=10)  # You can choose the number of components you want (e.g., 2 for 2D visualization)
    reduced_features = pca.fit_transform(pom_embeds)

    variance_explained = pca.explained_variance_ratio_

    # Variance explained by PC1 and PC2
    variance_pc1 = variance_explained[0]
    variance_pc2 = variance_explained[1]

    if is_preds:
        y = np.where(y_preds>threshold, 1.0, 0.0)
    else:
        y = dataset.y

    # Generate grid points to evaluate the KDE on
    x_grid, y_grid = np.meshgrid(np.linspace(reduced_features[:, 0].min(), reduced_features[:, 0].max(), 500),
                                 np.linspace(reduced_features[:, 1].min(), reduced_features[:, 1].max(), 500))
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])

    def get_kde_values(label):
        plot_idx = required_desc.index(label)
        label_indices = np.where(y[:, plot_idx] == 1)[0]
        kde_label = gaussian_kde(reduced_features[label_indices].T)
        kde_values_label = kde_label(grid_points)
        kde_values_label = kde_values_label.reshape(x_grid.shape)
        return kde_values_label
    
    def plot_contours(type_dictionary, bbox_to_anchor):
        main_label = list(type_dictionary.keys())[0]
        plt.contourf(x_grid, y_grid, get_kde_values(main_label), levels=1, colors=['#00000000',type_dictionary[main_label],type_dictionary[main_label]])
        legend_elements = []
        for label, color in type_dictionary['subs'].items():
            plt.contour(x_grid, y_grid, get_kde_values(label), levels=1, colors=color, linewidths=2)
            legend_elements.append(Patch(facecolor=color, label=label))
        legend = plt.legend(handles=legend_elements, title=main_label, bbox_to_anchor=bbox_to_anchor)
        legend.get_frame().set_facecolor(type_dictionary[main_label])
        plt.gca().add_artist(legend)

    plt.figure(figsize=(15, 10))
    plt.title('KDE Density Estimation with Contours in Reduced Space')
    plt.xlabel(f'Principal Component 1 ({round(variance_pc1*100, ndigits=2)}%)')
    plt.ylabel(f'Principal Component 2 ({round(variance_pc2*100, ndigits=2)}%)')
    plot_contours(type_dictionary=type1, bbox_to_anchor = (0.2, 0.8))
    plot_contours(type_dictionary=type2, bbox_to_anchor = (0.9, 0.4))
    plot_contours(type_dictionary=type3, bbox_to_anchor = (0.3, 0.1))
    # plt.colorbar(label='Density')
    # plt.show()
    png_file = os.path.join(dir, f'pom_frame_{epoch}.png')
    plt.savefig(png_file)
    plt.close()

