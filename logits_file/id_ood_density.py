# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde

# # Load the ID and OOD datasets
# id_conf = np.load('crl_id_conf.npy')
# ood_conf_places365 = np.load('crl_places365_ood_conf.npy')
# ood_conf_tin = np.load('crl_tin_ood_conf.npy')

# # Create a figure
# plt.figure()

# # Create a KDE plot for ID and OOD datasets
# kde_id = gaussian_kde(id_conf)
# kde_places365 = gaussian_kde(ood_conf_places365)
# kde_tin = gaussian_kde(ood_conf_tin)

# # Define the range of x values for the plot
# x_values = np.linspace(min(id_conf.min(), ood_conf_places365.min(), ood_conf_tin.min()),
#                        max(id_conf.max(), ood_conf_places365.max(), ood_conf_tin.max()), 1000)

# # Plot the KDEs for ID and OOD datasets
# plt.plot(x_values, kde_id(x_values), color='blue', label='ID Dataset')
# plt.plot(x_values, kde_places365(x_values), color='red', label='OOD Places365')
# #plt.plot(x_values, kde_tin(x_values), color='green', label='OOD TIN')

# # Fill the area below the KDE plot lines with color
# plt.fill_between(x_values, kde_id(x_values), color='blue', alpha=0.2)
# plt.fill_between(x_values, kde_places365(x_values), color='red', alpha=0.2)
# #plt.fill_between(x_values, kde_tin(x_values), color='green', alpha=0.2)

# # Add labels and legend
# plt.xlabel('Confidence Scores')
# plt.ylabel('Density')
# plt.legend(loc='upper right')

# # Save the plot as an image (e.g., in PNG format)
# plt.savefig('density_plot.png')

# # Show the plot (optional)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

# Load the ID and OOD datasets
postprocess_name = 'crl'        # crl, mls
id_conf_path = postprocess_name + '_' + 'id_conf.npy'
ood_conf_places365_path = postprocess_name + '_' + 'places365_ood_conf.npy'
ood_conf_tin_path = postprocess_name + '_' + 'tin_ood_conf.npy'
ood_conf_svhn_path = postprocess_name + '_' + 'svhn_ood_conf.npy'
ood_conf_texture_path = postprocess_name + '_' + 'texture_ood_conf.npy'


id_conf = np.load(id_conf_path)
ood_conf_places365 = np.load(ood_conf_places365_path)
ood_conf_tin = np.load(ood_conf_tin_path)
ood_conf_svhn = np.load(ood_conf_svhn_path)
ood_conf_texture = np.load(ood_conf_texture_path)

# Create a figure
plt.figure()

# Create a KDE plot for ID and OOD datasets
kde_id = gaussian_kde(id_conf)
kde_places365 = gaussian_kde(ood_conf_places365)
kde_tin = gaussian_kde(ood_conf_tin)
kde_svhn = gaussian_kde(ood_conf_svhn)
kde_texture = gaussian_kde(ood_conf_texture)

# Define the range of x values for the plot
x_values = np.linspace(min(id_conf.min(), ood_conf_places365.min(), ood_conf_tin.min()),
                       max(id_conf.max(), ood_conf_places365.max(), ood_conf_tin.max()), 1000)


light_green = (151/255, 209/255, 160/255)  # Custom color in RGB [0, 1] range
light_blue = (171/255, 218/255, 236/255)  # Custom color in RGB [0, 1] range


# Plot the KDEs for ID and OOD datasets
plt.plot(x_values, kde_id(x_values), color=light_green, label='ID')
plt.plot(x_values, kde_texture(x_values), color=light_blue, label='OOD (Texture)')
# plt.plot(x_values, kde_tin(x_values), color=light_blue, label='OOD (TinyImageNet)')

# Fill the area below the KDE plot lines with light blue and light green
plt.fill_between(x_values, kde_id(x_values), color=light_green, alpha=0.5)
plt.fill_between(x_values, kde_texture(x_values), color=light_blue, alpha=0.5)
# plt.fill_between(x_values, kde_tin(x_values), color=light_blue, alpha=0.5)

# Create a custom legend element as a colored patch
custom_color = (151/255, 209/255, 160/255)  # Custom color in RGB [0, 1] range

custom_legend_element = Patch(facecolor=custom_color, alpha=0.5)
# Create a legend with the custom legend element
plt.legend([custom_legend_element], ['Custom Area'], loc='upper right', frameon=False)

# Add labels and legend
plt.xlabel('Confidence Scores')
plt.ylabel('Density')
# plt.legend(loc='upper right')

# Save the plot as an image (e.g., in PNG format)
img_path = postprocess_name + '_density_plot.png'
plt.savefig(img_path)

# Show the plot (optional)
plt.show()
