import SimpleITK as sitk
from SimpleITK import CastImageFilter
from software_archive.localisation import alter_points_to_match_localised_region, generate_delineation
import numpy as np


class Experiment:
    def __init__(
        self,

        modality,
        data_path="/data",

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


    def run_full_slic_pipeline_on_img(self, img, pts):
        """
        Given an MR volume, perform the full localised SLIC superpixel segmentation algorithm.

        :param img: Original MR volume (full-size)
        :param pts: Centreline points
        :return: Weak label segmentation of the T.I.
        """

        # cropping
        index, size = generate_delineation(img, pts, padding=self.padding)
        cropped_img = sitk.RegionOfInterest(img, size=size, index=index)
        points = alter_points_to_match_localised_region(pts, index)

        # Create the filter of CastImageFilter
        myFilterCastImage = CastImageFilter()

        # Set output pixel type (float32)
        myFilterCastImage.SetOutputPixelType(sitk.sitkFloat32)
        cropped_img_float32 = myFilterCastImage.Execute(cropped_img)

        # Apply N4 bias field correction to the image
        n4_correction = sitk.N4BiasFieldCorrectionImageFilter()
        n4_correction.SetConvergenceThreshold(self.n4_convergence_threshold)
        n4_correction.SetMaximumNumberOfIterations(self.n4_max_num_of_iterations)
        n4_correction.SetBiasFieldFullWidthAtHalfMaximum(self.n4_bias_field_full_width_at_half_max)
        n4_correction.SetWienerFilterNoise(self.n4_wiener_filter_noise)
        n4_correction.SetNumberOfHistogramBins(self.n4_num_of_histogram_bins)
        n4_correction.SetNumberOfControlPoints(self.n4_num_of_control_points)
        n4_correction.SetSplineOrder(self.n4_spline_order)
        corrected_img = n4_correction.Execute(cropped_img_float32)

        # Denoising using curvature driven flow
        cflow = sitk.CurvatureFlowImageFilter()
        cflow.SetTimeStep(self.cflow_timestep)
        cflow.SetNumberOfIterations(self.cflow_num_of_iterations)
        denoised_img = cflow.Execute(corrected_img)

        lp_sharp = sitk.LaplacianSharpeningImageFilter()
        sharpened_edges_image = lp_sharp.Execute(denoised_img)

        # SLIC
        slic = sitk.SLICImageFilter()
        slic.SetMaximumNumberOfIterations(self.slic_max_num_of_iterations)
        slic.SetSuperGridSize(self.slic_super_grid_size)
        slic.SetSpatialProximityWeight(self.slic_spatial_prox_weight)
        slic.SetEnforceConnectivity(self.slic_enforce_connectivity)
        slic.SetInitializationPerturbation(self.slic_initialization_perturbation)
        seg = slic.Execute(sharpened_edges_image)

        out = get_slic_segments_using_coordinates(seg, points)
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

        return cropped_img, seg
    
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