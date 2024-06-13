import math
import os
import re
import xml.etree.ElementTree as ET
import SimpleITK as sitk
import numpy as np
from SimpleITK import CastImageFilter


class Experiment:
    def __init__(
        self,

        modality,
        data_path="data/pseudo_data/axial",

        task_num=None,
        target_task_num=None,

        write_results=True,
        write_segmentations=True,

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

        # SLIC parameters
        slic_max_num_of_iterations=5,
        slic_super_grid_size=(50, 50, 50),
        slic_spatial_prox_weight=10.0,
        slic_enforce_connectivity=True,
        slic_initialization_perturbation=True,

        # Voting iterative binary hole filling parameters
        voting_ibhole_filling_majority_threshold=1,
        voting_ibhole_filling_radius=(1, 1, 1),
        voting_ibhole_filling_max_num_of_iterations=10,

        # Binary morphologhical hole closing parameters
        morph_closing_safe_border=True,
        morph_closing_kernel_type=sitk.sitkBall,
        morph_closing_kernel_radius=(1, 1, 1),

        # Centreline parameters
        centreline_percentage=40,

        # Cropping parameters
        cropped=True,
        padding=15,

        experiments_path=None,
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
        self.slic_max_num_of_iterations = slic_max_num_of_iterations
        self.slic_super_grid_size = slic_super_grid_size
        self.slic_spatial_prox_weight = slic_spatial_prox_weight
        self.slic_enforce_connectivity = slic_enforce_connectivity
        self.slic_initialization_perturbation = slic_initialization_perturbation
        self.voting_ibhole_filling_majority_threshold = voting_ibhole_filling_majority_threshold
        self.voting_ibhole_filling_radius = voting_ibhole_filling_radius
        self.voting_ibhole_filling_max_num_of_iterations = voting_ibhole_filling_max_num_of_iterations
        self.morph_closing_safe_border = morph_closing_safe_border
        self.morph_closing_kernel_type = morph_closing_kernel_type
        self.morph_closing_kernel_radius = morph_closing_kernel_radius
        self.padding = padding
        self.centreline_percentage = centreline_percentage

        # Other parameters
        self.data_path = data_path
        self.modality = modality
        self.task_num = task_num
        self.target_task_num = target_task_num
        self.start_time = None
        self.experiments_path = "./"
        self.write_results = write_results
        self.write_segmentations = write_segmentations

        # Overlap measures
        self.dice_coefficients = []
        self.false_negative_errors = []
        self.false_positive_errors = []
        self.jaccard_coefficients = []
        self.mean_overlaps = []
        self.union_overlaps = []
        self.volume_similarities = []

        # Cropped or not
        self.cropped = cropped


    def run_full_uncropped_slic_pipeline_on_case(self, case_name, img=None, pts=None):
        """
        Run the full superpixel segmentation algorithm, but without localisation, on a specific
        case.

        :param case_name: The name of the case, e.g. 'A001_axial'
        :param img: Original MR volume (full-size)
        :param pts: Centreline points
        :return: The MR scan, generated segmentation, and corresponding ground-truth segmentation
        """
        # MR scan
        if img is None:
            img_path = os.path.join(self.data_path, f'img/{case_name}.nii.gz')
            img = sitk.ReadImage(img_path)
        
        # Centreline points
        if pts is None:
            pts_path = os.path.join(self.data_path, f'centreline/{case_name} HASTE tracings')
            if not os.path.exists(pts_path):
                return None
            pts = get_centreline_points_from_xml(pts_path, percentage=self.centreline_percentage)

        # Create the filter of CastImageFilter
        myFilterCastImage = CastImageFilter()

        # Set output pixel type (float32)
        myFilterCastImage.SetOutputPixelType(sitk.sitkFloat32)
        img_float32 = myFilterCastImage.Execute(img)

        # Apply N4 bias field correction to the image
        n4_correction = sitk.N4BiasFieldCorrectionImageFilter()
        n4_correction.SetConvergenceThreshold(self.n4_convergence_threshold)
        n4_correction.SetMaximumNumberOfIterations(self.n4_max_num_of_iterations)
        n4_correction.SetBiasFieldFullWidthAtHalfMaximum(self.n4_bias_field_full_width_at_half_max)
        n4_correction.SetWienerFilterNoise(self.n4_wiener_filter_noise)
        n4_correction.SetNumberOfHistogramBins(self.n4_num_of_histogram_bins)
        n4_correction.SetNumberOfControlPoints(self.n4_num_of_control_points)
        n4_correction.SetSplineOrder(self.n4_spline_order)
        corrected_img = n4_correction.Execute(img_float32)

        lp_sharp = sitk.LaplacianSharpeningImageFilter()
        sharpened_edges_image = lp_sharp.Execute(corrected_img)

        # Denoising using curvature driven flow
        cflow = sitk.CurvatureFlowImageFilter()
        cflow.SetTimeStep(self.cflow_timestep)
        cflow.SetNumberOfIterations(self.cflow_num_of_iterations)
        denoised_img = cflow.Execute(sharpened_edges_image)

        # SLIC
        slic = sitk.SLICImageFilter()
        slic.SetMaximumNumberOfIterations(self.slic_max_num_of_iterations)
        slic.SetSuperGridSize(self.slic_super_grid_size)
        slic.SetSpatialProximityWeight(self.slic_spatial_prox_weight)
        slic.SetEnforceConnectivity(self.slic_enforce_connectivity)
        slic.SetInitializationPerturbation(self.slic_initialization_perturbation)
        seg = slic.Execute(denoised_img)

        out = get_slic_segments_using_coordinates(seg, pts)
        out.CopyInformation(seg)

        # Do voting binary hole filling
        voting_ibhole_filling = sitk.VotingBinaryHoleFillingImageFilter()
        voting_ibhole_filling.SetBackgroundValue(0.0)
        voting_ibhole_filling.SetForegroundValue(1.0)
        voting_ibhole_filling = sitk.VotingBinaryIterativeHoleFillingImageFilter()
        voting_ibhole_filling.SetRadius(self.voting_ibhole_filling_radius)
        voting_ibhole_filling.SetMaximumNumberOfIterations(self.voting_ibhole_filling_max_num_of_iterations)
        voting_ibhole_filling.SetMajorityThreshold(self.voting_ibhole_filling_majority_threshold)
        seg_after_binary_hole_filling = voting_ibhole_filling.Execute(out)

        # Do morphological hole closing
        morph_closing = sitk.BinaryMorphologicalClosingImageFilter()
        morph_closing.SetSafeBorder(self.morph_closing_safe_border)
        morph_closing.SetKernelRadius(self.morph_closing_kernel_radius)
        morph_closing.SetKernelType(self.morph_closing_kernel_type)
        seg = morph_closing.Execute(seg_after_binary_hole_filling)

        return seg


def get_slic_segments_using_coordinates(seg: sitk.Image, points: list):
    """
    Using a SLIC-segmented image and centreline co-ordinates, select the superpixel clusters which
    lie on the centreline. 
    
    :param seg: SLIC-segmented SimpleITK image
    :param points: list of centreline co-ordinates
    :return: selected superpixel clusters, as a SimpleITK image
    """
    def apply_intensity_mask(arr, intensity_mask):
        arr[intensity_mask] = 0

    def generate_intensity_mask(arr, required_intensities):
        return ~np.isin(arr, required_intensities)

    required_intensities = []
    arr = sitk.GetArrayFromImage(seg)

    for point in points:
        x, y, z = point
        # numpy requires (z, y, x) form
        intensity = arr[int(z), int(y), int(x)]
        if intensity not in required_intensities:
            required_intensities.append(intensity)

    intensity_mask = generate_intensity_mask(arr, required_intensities)
    apply_intensity_mask(arr, intensity_mask)

    return sitk.GetImageFromArray(arr)


def get_centreline_points_from_xml(xml_file_path, percentage=100, keep_normal_pts=True, keep_abnormal_pts=True):
    """
    Get a list of points [(x, y, z), ...] from an XML traces file provided by the clinicians.
    :param xml_file_path: Path to the XML file
    :param percentage: Percentage of the centreline to take into account (T.I.=20%)
    :param keep_normal_pts: Whether to keep the colon co-ordinates classified as being a normal region
    :param keep_abnormal_pts: Whether to keep the colon co-ordinates classified as being an abnormal region
    :return: List of centreline points in the format [(x, y, z), ...]
    """
    
    out = []
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    for path in root:
        if 'name' not in path.attrib:
            continue
        
        # Check if the co-ordinate lies on an abnormal region of the colon
        is_abnormal_path = re.search('abnormal', path.attrib['name'], re.IGNORECASE)
        is_normal_path = not is_abnormal_path

        if is_abnormal_path and (not keep_abnormal_pts):
            continue

        if is_normal_path and (not keep_normal_pts):
            continue

        for point in path:
            attr = point.attrib
            out.append((int(attr['x']), int(attr['y']), int(attr['z'])))

    size = int(math.ceil(len(out) * percentage / 100))
    return out[:size]


if __name__ == "__main__":
    exp =  Experiment(
    task_num=501,
    modality='axial',

    write_results=False,
    write_segmentations=False,
    
    # N4 Bias Field Parameters
    n4_convergence_threshold=0.001,
    n4_max_num_of_iterations=(10, 10, 10, 10),
    n4_bias_field_full_width_at_half_max=0.1,
    n4_wiener_filter_noise=0.01,
    n4_num_of_histogram_bins=100,
    n4_num_of_control_points=(4, 4, 4),
    n4_spline_order=3,

    # Denoising (curvature flow) parameters
    cflow_timestep=0.005,
    cflow_num_of_iterations=100,

    # SLIC superpixel segmentation parameters
    slic_max_num_of_iterations=50,
    slic_super_grid_size=(6, 6, 6),
    slic_spatial_prox_weight=5.0,
    slic_enforce_connectivity=True,
    slic_initialization_perturbation=True,

    # Voting iterative binary hole filling parameters
    voting_ibhole_filling_majority_threshold=1,
    voting_ibhole_filling_radius=(2, 2, 2),
    voting_ibhole_filling_max_num_of_iterations=50,

    # Binary morphologhical hole closing parameters
    morph_closing_safe_border=True,
    morph_closing_kernel_type=sitk.sitkBox,
    morph_closing_kernel_radius=(7, 7, 7),

    # Cropping parameters
    padding=20,

    # Centreline parameters
    centreline_percentage=20,
)   
    
    write_path = "data/pseudo_data/axial/seg"
    for file in sorted(os.listdir("data/pseudo_data/axial/img")):
        if file.endswith(".nii.gz"):
            case_name = file.split(".")[0]
            print(case_name)
            seg = exp.run_full_uncropped_slic_pipeline_on_case(case_name)
            if seg is None:
                print("No centreline points are given. Skipping...")
                continue
            sitk.WriteImage(seg, os.path.join(write_path, f"{case_name}.nii.gz"))
            print("Done")
