import numpy as np
import matplotlib.pyplot as plt

# Load the ID and OOD datasets
postprocess_name = 'crl'
id_conf_path = postprocess_name + '_' + 'id_conf.npy'
ood_conf_places365_path = postprocess_name + '_' + 'places365_ood_conf.npy'
ood_conf_tin_path = postprocess_name + '_' + 'tin_ood_conf.npy'
ood_conf_svhn_path = postprocess_name + '_' + 'svhn_ood_conf.npy'
ood_conf_texture_path = postprocess_name + '_' + 'texture_ood_conf.npy'

# Load the ID and OOD datasets
id_conf = np.load(id_conf_path)
ood_conf_places365 = np.load(ood_conf_places365_path)
ood_conf_tin = np.load(ood_conf_tin_path)
ood_conf_svhn = np.load(ood_conf_svhn_path)
ood_conf_texture = np.load(ood_conf_texture_path)

# Create a figure
plt.figure()
light_green = (151/255, 209/255, 160/255)   # Custom color in RGB [0, 1] range
light_blue = (171/255, 218/255, 236/255)    # Custom color in RGB [0, 1] range
light_red = (253/255, 213/255, 192/255)     # Custom color in RGB [0, 1] range

# Create histograms for ID and OOD datasets
plt.hist(id_conf, bins=50, color=light_green, alpha=0.6, label='ID')
plt.hist(ood_conf_tin, bins=50, color=light_red, alpha=0.6, label='OOD (TinyImageNet)')
# plt.hist(ood_conf_tin, bins=50, color='green', alpha=0.5, label='OOD TIN')

# Add labels and legend
plt.xlabel('Confidence Scores')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# Save the plot as an image (e.g., in PNG format)
img_path = postprocess_name + '_density_plot.png'
plt.savefig(img_path)

# Show the plot (optional)
plt.close()


