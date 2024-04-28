import SimpleITK as sitk
from SimpleITK import CastImageFilter
import os
import shutil
import numpy as np
import re
import tqdm
import matplotlib.pyplot as plt
from prompt_generator import get_centreline_points_from_file,centreline_prompt
import pandas as pd
from utils.display_helper import show_box,show_mask,show_points
from skimage.morphology import skeletonize_3d,remove_small_objects


class Preprocessor:
    def __init__(
        self,
        # N4 Bias Field Parameters
        n4_convergence_threshold=0.001,
        n4_max_num_of_iterations=(50, 50, 50, 50),
        n4_bias_field_full_width_at_half_max=0.15,
        n4_wiener_filter_noise=0.01,
        n4_num_of_histogram_bins=200,
        n4_num_of_control_points=(4, 4, 4),
        n4_spline_order=3,

        # Denoising (curvature flow) parameters
        cflow_timestep=0.05,
        cflow_num_of_iterations=5,

    ):
        # Hyper-parameters
        self.n4_convergence_threshold = n4_convergence_threshold
        self.n4_max_num_of_iterations = n4_max_num_of_iterations
        self.n4_bias_field_full_width_at_half_max = n4_bias_field_full_width_at_half_max
        self.n4_wiener_filter_noise = n4_wiener_filter_noise
        self.n4_num_of_histogram_bins = n4_num_of_histogram_bins
        self.n4_num_of_control_points = n4_num_of_control_points
        self.n4_spline_order = n4_spline_order
        self.cflow_timestep = cflow_timestep
        self.cflow_num_of_iterations = cflow_num_of_iterations

        
    def preprocess(self, img):
        # Create the filter of CastImageFilter
        myFilterCastImage = CastImageFilter()

        # Set output pixel type (float32)
        myFilterCastImage.SetOutputPixelType(sitk.sitkFloat32)
        input_image = myFilterCastImage.Execute(img)

        # Apply N4 bias field correction to the image
        n4_correction = sitk.N4BiasFieldCorrectionImageFilter()
        n4_correction.SetConvergenceThreshold(self.n4_convergence_threshold)
        n4_correction.SetMaximumNumberOfIterations(self.n4_max_num_of_iterations)
        n4_correction.SetBiasFieldFullWidthAtHalfMaximum(self.n4_bias_field_full_width_at_half_max)
        n4_correction.SetWienerFilterNoise(self.n4_wiener_filter_noise)
        n4_correction.SetNumberOfHistogramBins(self.n4_num_of_histogram_bins)
        n4_correction.SetNumberOfControlPoints(self.n4_num_of_control_points)
        n4_correction.SetSplineOrder(self.n4_spline_order)
        corrected_img = n4_correction.Execute(input_image)

        # Denoising using curvature driven flow
        cflow = sitk.CurvatureFlowImageFilter()
        cflow.SetTimeStep(self.cflow_timestep)
        cflow.SetNumberOfIterations(self.cflow_num_of_iterations)
        denoised_img = cflow.Execute(corrected_img)

        # Laplacian sharpening
        lp_sharp = sitk.LaplacianSharpeningImageFilter()
        sharpened_edges_image = lp_sharp.Execute(denoised_img)
        return sharpened_edges_image


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
        

class Generate_centreline_points:
    """
    Generate centreline points from the segmentation mask.
    """
    def __init__(self, seg_folder, output_folder, margin=5):
        self.seg_folder = seg_folder
        self.output_folder = output_folder
        self.margin = margin  # Margin to exclude points near the segment edges

    def generate_centreline(self):
        """
        Generate centreline points from the segmentation mask.
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
        centreline_points = []
        for slice_idx in tqdm.tqdm(range(seg_arr.shape[0])):
            slice = seg_arr[slice_idx]
            points = self._find_centreline_points(slice)
            for point in points:
                centreline_points.append((slice_idx, point[0], point[1]))
        return centreline_points

    def _find_centreline_points(self, slice):
        centreline_points = []
        for row_idx, row in enumerate(slice):
            points = self._find_centreline_points_in_row(row)
            centreline_points.extend([(row_idx, col_idx) for col_idx in points])
        return centreline_points

    def _find_centreline_points_in_row(self, row):
        """
        Locate the center points of contiguous segments of white (1's) in a row,
        excluding the margins at both ends of each segment.
        """
        centre_points = []
        in_segment = False
        start = 0

        for i, val in enumerate(row):
            if val == 1 and not in_segment:
                start = i
                in_segment = True
            elif val == 0 and in_segment:
                end = i
                # Apply the margin to the start and end
                segment_center = (max(start + self.margin, 0) + min(end - self.margin, len(row))) // 2
                if max(start + self.margin, 0) < min(end - self.margin, len(row)):
                    centre_points.append(segment_center)
                in_segment = False
        if in_segment:
            end = len(row)
            segment_center = (max(start + self.margin, 0) + min(end - self.margin, len(row))) // 2
            if max(start + self.margin, 0) < min(end - self.margin, len(row)):
                centre_points.append(segment_center)

        return centre_points


    """
    Generate centreline points from the segmentation mask.
    """

    def __init__(self, seg_folder, output_folder):
        self.seg_folder = seg_folder
        self.output_folder = output_folder

    def generate_centreline(self):
        """
        Generate centreline points from the 3D segmentation mask using skeletonization.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        for filename in tqdm.tqdm(os.listdir(self.seg_folder)):
            seg_path = os.path.join(self.seg_folder, filename)
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg)
            seg_skeleton = skeletonize_3d(seg_arr)
            seg_skeleton = self._prune_skeleton(seg_skeleton, 10)  # adjust min_size as needed

            centreline_points = self._extract_centreline_points(seg_skeleton)
            output_filename = filename.replace('.nii.gz', '.txt')
            output_path = os.path.join(self.output_folder, output_filename)
            with open(output_path, 'w') as file:
                for point in centreline_points:
                    file.write(f'{point[0]} {point[1]} {point[2]}\n')
    def _prune_skeleton(seg_skeleton, min):
        """
        Remove small objects from the skeleton to reduce noise.
        """
        cleaned_skeleton = remove_small_objects(seg_skeleton, min_size=min, connectivity=3)
        return cleaned_skeleton
    def _extract_centreline_points(self, seg_skeleton):
        """
        Extract centreline points from the 3D skeletonized array.
        """
        centreline_points = []
        # Get the indices of non-zero elements (i.e., skeleton points)
        z_indices, y_indices, x_indices = np.nonzero(seg_skeleton)
        for z, y, x in zip(z_indices, y_indices, x_indices):
            centreline_points.append((z, y, x))
        return centreline_points

class Generate3DConnectCentrelinePoints:
    """
    Generate centreline points from the segmentation mask.
    """

    def __init__(self, seg_folder, output_folder):
        self.seg_folder = seg_folder
        self.output_folder = output_folder

    def generate_centreline(self):
        """
        Generate centreline points from the 3D segmentation mask using skeletonization.
        """
        os.makedirs(self.output_folder, exist_ok=True)
        for filename in tqdm.tqdm(os.listdir(self.seg_folder)):
            seg_path = os.path.join(self.seg_folder, filename)
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg)
            seg_skeleton = skeletonize_3d(seg_arr.astype(bool))  # ensure input is boolean
            seg_skeleton = self._prune_skeleton(seg_skeleton, 48)  # adjust min_size as needed

            centreline_points = self._extract_centreline_points(seg_skeleton)
            output_filename = filename.replace('.nii.gz', '.txt')
            output_path = os.path.join(self.output_folder, output_filename)
            with open(output_path, 'w') as file:
                for point in centreline_points:
                    file.write(f'{point[0]} {point[1]} {point[2]}\n')

    def _prune_skeleton(self, seg_skeleton, min_size):
        """
        Remove small objects from the skeleton to reduce noise.
        """
        cleaned_skeleton = remove_small_objects(seg_skeleton, min_size=min_size, connectivity=1)
        return cleaned_skeleton

    def _extract_centreline_points(self, seg_skeleton):
        """
        Extract centreline points from the 3D skeletonized array.
        """
        centreline_points = []
        # Get the indices of non-zero elements (i.e., skeleton points)
        z_indices, y_indices, x_indices = np.nonzero(seg_skeleton)
        for z, y, x in zip(z_indices, y_indices, x_indices):
            centreline_points.append((z, y, x))
        return centreline_points


if __name__ == "__main__":
    testing = 0
    display = 0

    if not testing and not display:
        generator = Generate3DConnectCentrelinePoints("data/coronal/seg", "data/coronal/centreline_3d")
        generator.generate_centreline()
        display = 1
        
    if testing and not display:
        for filename in tqdm.tqdm(sorted(os.listdir("data/coronal/centreline_3d"))):
            img_path = os.path.join("data/coronal/img", filename.replace('.txt', '.nii.gz'))
            seg_path = os.path.join("data/coronal/seg", filename.replace('.txt', '.nii.gz'))
            img = sitk.ReadImage(img_path)
            img_arr = sitk.GetArrayFromImage(img)
            seg = sitk.ReadImage(seg_path)
            seg_arr = sitk.GetArrayFromImage(seg)
            centreline_path = os.path.join("data/coronal/centreline", filename)
            points_df = get_centreline_points_from_file(centreline_path)

            for slice_number in range(seg_arr.shape[0]):  # Corrected iteration over slices
                img_slice = img_arr[slice_number]
                seg_slice = seg_arr[slice_number]
                points_in_slice = centreline_prompt(points_df, slice_number)
                points_label = np.ones((len(points_in_slice)))
                if len(points_in_slice) == 0:
                    continue
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.title(f"Image and Points - Slice {slice_number}")
                plt.imshow(img_slice, cmap='gray')
                show_points(points_in_slice, points_label, plt.gca(), marker_size=50)
                plt.subplot(1, 2, 2)
                plt.title(f"Segmentation - Slice {slice_number}")
                plt.imshow(seg_slice, cmap='gray')
                show_mask(seg_slice, plt.gca())
                save_path = os.path.join("weak_3dcentreline_visualise", filename.replace('.txt', ''), f"slice_{slice_number}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
                plt.close()
                    
    if display:
        for filename in tqdm.tqdm(os.listdir('data/coronal/centreline_3d')):
            data_path = os.path.join('data/coronal/centreline_3d', filename)
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
            saved_path = os.path.join('centreline_plot', filename.replace('.txt', '.png'))
            
            if not os.path.exists(os.path.dirname(saved_path)):
                os.makedirs(os.path.dirname(saved_path))
            
            plt.savefig(saved_path, bbox_inches='tight', dpi=200)
            plt.close()
                
    