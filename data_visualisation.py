import SimpleITK as sitk
import matplotlib.pyplot as plt
import os 

def read_image(image_path):
    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image)
    return image


def save_figure(image, save_path, slice, title=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(f"{save_path}/slice_{slice}.png", bbox_inches='tight', dpi=200)
    plt.close()
    
if __name__ ==  "__main__":
    image_path = "data/contrast/img/A113 contrast.nii.gz"
    image = read_image(image_path)
    save_path = "data_visualisation/A113_contrast"
    for i in range(image.shape[0]):
        save_figure(image[i], save_path, i,title=f"A113 contrast Slice {i}")