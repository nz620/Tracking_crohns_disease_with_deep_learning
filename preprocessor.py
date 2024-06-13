import SimpleITK as sitk
import os
import shutil
import numpy as np
import re
import tqdm
import matplotlib.pyplot as plt
from prompt_generator import get_centreline_points_from_file,centreline_prompt
from utils.display_helper import show_mask,show_points
from skimage.measure import label
from skimage.morphology import medial_axis,label


def overwrite_labels(src_folder, target_folder, OnlyTI=False, OnlyATI=False):
    """
    Overwrite the labels of segmentations in a folder and save them to a target folder.
    
    :param src_folder: Folder containing the original segmentations.
    :param target_folder: Directory to save the relabelled segmentations.
    :param OnlyTI: Only set the TI area as foreground
    :param OnlyATI: Only set the abnormal TI area as foreground
    :return: None
    """
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    # Iterate over all files in the source folder
    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)
        # Read the source image and segmentation
        
        seg = sitk.ReadImage(file_path)
        arr = sitk.GetArrayFromImage(seg)

        # Apply the relabeling logic
        arr = np.where(arr == 6, 0, arr) # appendix 
        arr = np.where(arr == 3, 0, arr) # colon
        arr = np.where(arr == 4, 0, arr) # colon
        if OnlyTI:
            arr = np.where(arr == 1, 1, arr) # abnormal terminal ileum
            arr = np.where(arr == 2, 1, arr) # normal terminal ileum
        if OnlyATI:
            arr = np.where(arr == 1, 1, arr) # abnormal terminal ileum
            arr = np.where(arr == 2, 0, arr) # normal terminal ileum

        # Create new segmentation image
        seg_new = sitk.GetImageFromArray(arr)
        if seg_new.GetSize() == seg.GetSize():
            seg_new.CopyInformation(seg)

        # Save the new segmentation
        new_file_path = os.path.join(target_folder, filename)
        sitk.WriteImage(seg_new, new_file_path)
        
def tidy_folder(source_folder,target_folder):
    # Paths for the 'seg' and 'image' subfolders
    seg_folder = os.path.join(target_folder, 'seg')
    image_folder = os.path.join(target_folder, 'img')

    # Create the subfolders if they don't exist
    os.makedirs(seg_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    # Iterate over all files in the source folder
    for filename in os.listdir(source_folder):
            file_path = os.path.join(source_folder, filename)

            # Skip directories
            if os.path.isdir(file_path):
                continue

            # Remove spaces in the filename
            new_filename = filename.replace(" ", "")

            # Move file to the appropriate folder based on its name
            if 'seg' in new_filename.lower():
                shutil.move(file_path, os.path.join(seg_folder, new_filename))
            else:
                shutil.move(file_path, os.path.join(image_folder, new_filename))

def rename_files(directory):
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern (e.g., A101axial.npz)
        match = re.match(r'(A\d+)([^\.]+)\.nii', filename) or \
                re.match(r'(A\d+)([^\.]+)\.nii.gz', filename) or \
                re.match(r'(a\d+)([^\.]+)\.nii', filename) or \
                re.match(r'(a\d+)([^\.]+)\.nii.gz', filename) or \
                re.match(r'(I\d+)([^\.]+)\.nii', filename) or \
                re.match(r'(I\d+)([^\.]+)\.nii.gz', filename) or \
                re.match(r'(i\d+)([^\.]+)\.nii', filename) or \
                re.match(r'(i\d+)([^\.]+)\.nii.gz', filename) 
        if match:
            id_part = match.group(1).upper()  # Capture 'A101'
            plane_part = match.group(2)  # Capture 'axial' or similar
            # Simplify the plane part
            new_plane_part = 'unknown'
            if 'contrast' in plane_part.lower():
                new_plane_part = 'contrast'
            elif 'axial' in plane_part.lower():
                new_plane_part = 'axial'
            elif 'coronal' or 'cor' in plane_part.lower():
                new_plane_part = 'coronal'
            # Construct the new filename
            new_filename = f"{id_part} {new_plane_part}.nii.gz"
            # Rename the file
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f'Renamed "{filename}" to "{new_filename}"')
        else:
            print(f'Filename "{filename}" does not match the pattern, skipped.')

def tidy_plane(source_directory, target_directory, type='img'):
    # Iterate over all files in the source directory
    for filename in os.listdir(source_directory):
        sub_target_directory = target_directory  # Use a separate variable for subdirectories

        if 'axial' in filename.lower():
            sub_target_directory = os.path.join(sub_target_directory, 'axial', type)
        elif 'coronal' in filename.lower():
            sub_target_directory = os.path.join(sub_target_directory, 'coronal', type)
        elif 'contrast' in filename.lower():
            sub_target_directory = os.path.join(sub_target_directory, 'contrast', type)
        else:
            continue  # Skip file if it doesn't contain any of the keywords

        if not os.path.exists(sub_target_directory):
            os.makedirs(sub_target_directory)

        source_path = os.path.join(source_directory, filename)
        target_path = os.path.join(sub_target_directory, filename)
        shutil.copy2(source_path, target_path)
        print(f'Copied "{filename}" to "{sub_target_directory}"')
        


# class Generate_single_centreline_points:
#     """
#     Generate a single centreline point from the segmentation mask, which represents the centroid of the segment.
#     """
#     def __init__(self, seg_folder, output_folder):
#         self.seg_folder = seg_folder
#         self.output_folder = output_folder

#     def generate_centreline(self):
#         """
#         Generate centreline points from the segmentation mask.
#         """
#         os.makedirs(self.output_folder, exist_ok=True)
#         for filename in os.listdir(self.seg_folder):
#             seg_path = os.path.join(self.seg_folder, filename)
#             seg = sitk.ReadImage(seg_path)
#             seg_arr = sitk.GetArrayFromImage(seg)

#             centreline_points = self._generate_centreline_points(seg_arr)

#             output_filename = filename.replace('.nii.gz', '.txt')
#             output_path = os.path.join(self.output_folder, output_filename)
#             with open(output_path, 'w') as file:
#                 for point in centreline_points:
#                     file.write(f'{point[0]} {point[1]} {point[2]}\n')

#     def _generate_centreline_points(self, seg_arr):
#         """
#         Generate the centre point for each connected component in the segmentation array.
#         """
#         centreline_points = []
#         for slice_idx in tqdm.tqdm(range(seg_arr.shape[0])):
#             slice = seg_arr[slice_idx]
#             labeled_image = sitk.ConnectedComponent(sitk.GetImageFromArray(slice))
            
#             # Use LabelShapeStatisticsImageFilter to find centroids
#             label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
#             label_shape_filter.Execute(labeled_image)
#             num_features = label_shape_filter.GetNumberOfLabels()

#             if num_features > 0:
#                 for label in range(1, num_features + 1):
#                     # Find the centroid of each label
#                     centroid = label_shape_filter.GetCentroid(label)
#                     centreline_points.append((slice_idx, int(centroid[1]), int(centroid[0])))  # Ensure x, y are in correct order

#         return centreline_points

class GenerateCentrelinePoints:
    """
    Generate a single centreline point from each segmentation within the mask, representing the centroid of the segment.
    """
    def __init__(self, seg_folder, output_folder):
        self.seg_folder = seg_folder
        self.output_folder = output_folder

    def generate_centreline(self):
        """
        Generate centreline points from the segmentation mask for each disconnected component.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        for filename in os.listdir(self.seg_folder):
            seg_path = os.path.join(self.seg_folder, filename)
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg)

            centreline_points = self._generate_centreline_points(seg_arr)

            output_filename = filename.replace('.nii.gz', '.txt')
            output_path = os.path.join(self.output_folder, output_filename)
            with open(output_path, 'w') as file:
                for point in centreline_points:
                    file.write(f'{point[0]} {point[1]} {point[2]}\n')

    def _generate_centreline_points(self, seg_arr):
        """
        Generate the centre points using the medial axis for each connected component in the segmentation array.
        """
        centreline_points = []
        for slice_idx in tqdm.tqdm(range(seg_arr.shape[0])):
            slice = seg_arr[slice_idx]
            if np.any(slice): 
                labeled_slice, num_features = label(slice, return_num=True, connectivity=2)

                for i in range(1, num_features + 1):  # Process each component
                    component = labeled_slice == i
                    skeleton, distance = medial_axis(component, return_distance=True)
                    
                    max_dist = distance.max()
                    central_points = np.argwhere(distance == max_dist)

                    if central_points.size > 0:
                        idx = len(central_points)//2
                        central_point = central_points[idx]
                        centreline_points.append((slice_idx, int(central_point[0]), int(central_point[1])))

        return centreline_points

# class GenerateCentrelinePoints:
#     def __init__(self, seg_folder, output_folder, viz_folder):
#         self.seg_folder = seg_folder
#         self.output_folder = output_folder
#         self.viz_folder = viz_folder  # Folder to save visualizations

#     def generate_centreline(self):
#         os.makedirs(self.output_folder, exist_ok=True)
#         os.makedirs(self.viz_folder, exist_ok=True)
#         for filename in os.listdir(self.seg_folder):
#             seg_path = os.path.join(self.seg_folder, filename)
#             seg = sitk.ReadImage(seg_path)
#             seg_arr = sitk.GetArrayFromImage(seg)

#             centreline_points, visualizations = self._generate_centreline_points(seg_arr)

#             output_filename = filename.replace('.nii.gz', '.txt')
#             output_path = os.path.join(self.output_folder, output_filename)
#             with open(output_path, 'w') as file:
#                 for point in centreline_points:
#                     file.write(f'{point[0]} {point[1]} {point[2]}\n')
#             for idx, (viz, points) in enumerate(zip(visualizations, centreline_points)):
#                 fig, axs = plt.subplots(1, 3, figsize=(10, 6))
#                 show_mask(viz[0], axs[0])
#                 axs[0].axis('off')
                
#                 show_mask(viz[0], axs[1])
#                 show_mask(viz[1], axs[1],random_color=True)
#                 axs[1].axis('off')
                
#                 show_mask(viz[0], axs[2])
#                 axs[2].scatter(points[2], points[1], color='red', s=10)  # Plot the center point
#                 axs[2].axis('off')
#                 fig.tight_layout

#                 plt.savefig(os.path.join(self.viz_folder, f'{filename}_slice_{idx}.png'))
                
#                 plt.close()


#     def _generate_centreline_points(self, seg_arr):
#         centreline_points = []
#         visualizations = []
#         for slice_idx in tqdm.tqdm(range(seg_arr.shape[0])):
#             slice = seg_arr[slice_idx]
#             if np.any(slice):
#                 labeled_slice, num_features = label(slice, return_num=True, connectivity=2)
#                 for i in range(1, num_features + 1):
#                     component = labeled_slice == i
#                     skeleton, distance = medial_axis(component, return_distance=True)
                    
#                     max_dist = distance.max()
#                     central_points = np.argwhere(distance == max_dist)

#                     if central_points.size > 0:
#                         idx = len(central_points) // 2
#                         central_point = central_points[idx]
#                         centreline_points.append((slice_idx, central_point[0], central_point[1]))

#                         # Visualization: combine component, skeleton and center point
#                         visualization = [component.astype(float), skeleton.astype(float), np.zeros_like(component, dtype=float)]
#                         visualizations.append(visualization)
#         return centreline_points, visualizations

if __name__ == "__main__":
    testing = 0
    display = 0

    if not testing and not display:
        # generator = Generate_single_centreline_points("data/centreline_set/axial/seg", "data/centreline_set/axial/centreline_single")
        generator = GenerateCentrelinePoints("data/axial/seg", "data/axial/centreline_single_skeltonize")
        generator.generate_centreline()
        testing = 0
        
    if testing and not display:
        for filename in tqdm.tqdm(sorted(os.listdir("data/centreline_set/axial/centreline_single_skeltonize"))):
            img_path = os.path.join("data/centreline_set/axial/img", filename.replace('.txt', '.nii.gz'))
            seg_path = os.path.join("data/centreline_set/axial/seg", filename.replace('.txt', '.nii.gz'))
            img = sitk.ReadImage(img_path)
            img_arr = sitk.GetArrayFromImage(img)
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg)
            centreline_path = os.path.join("data/centreline_set/axial/centreline_single_skeltonize", filename)
            points_df = get_centreline_points_from_file(centreline_path)
            real_centreline_points = get_centreline_points_from_file(os.path.join("data/centreline_set/axial/centreline", filename.replace('.txt', ' HASTE tracings')))
            for slice_number in range(seg_arr.shape[0]):  # Corrected iteration over slices
                img_slice = img_arr[slice_number]
                seg_slice = seg_arr[slice_number]
                points_in_slice = centreline_prompt(points_df, slice_number)
                points_label = np.zeros((len(points_in_slice)))  
                real_points_in_slice = centreline_prompt(real_centreline_points, slice_number)
                real_points_in_slice_label = np.ones((len(real_points_in_slice)))
                if len(points_in_slice) == 0:
                    continue
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.title(f"Image and Points - Slice {slice_number}")
                plt.imshow(img_slice, cmap='gray')
                show_points(points_in_slice, points_label, plt.gca(), marker_size=50)
                show_points(real_points_in_slice, real_points_in_slice_label, plt.gca(), marker_size=50)
                plt.subplot(1, 2, 2)
                plt.title(f"Segmentation - Slice {slice_number}")
                plt.imshow(seg_slice, cmap='gray')
                show_mask(seg_slice, plt.gca())
                save_path = os.path.join("weak_complete_centreline_axial_single_skeletonize_visualise", filename.replace('.txt', ''), f"slice_{slice_number}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
                plt.close()
    

                    
    if display:
        for filename in tqdm.tqdm(os.listdir('data/centreline_set/axial/centreline_single_skeltonize')):
            data_path = os.path.join('data/centreline_set/axial/centreline_single_skeltonize', filename)
            centreline_points = get_centreline_points_from_file(data_path, percentage=20)

            xs = centreline_points["x"]
            ys = centreline_points["y"]
            zs = centreline_points["z"]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs, color='red')

            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Slice Number')
            ax.set_title('3D Scatter Plot of Centreline Points')
            saved_path = os.path.join('centreline_single_skeltonize_axial_plot', filename.replace('.txt', '.png'))
            
            if not os.path.exists(os.path.dirname(saved_path)):
                os.makedirs(os.path.dirname(saved_path))
            
            plt.savefig(saved_path, bbox_inches='tight', dpi=200)
            plt.close()
    # tidy_plane("data/pseudo_data/centreline", "data/pseudo_data", type='centreline')
    # tidy_plane("data/centreline_set/segATI", "data/centreline_set", type='segATI')
    # rename_files("data/pseudo_data/img")

    
    