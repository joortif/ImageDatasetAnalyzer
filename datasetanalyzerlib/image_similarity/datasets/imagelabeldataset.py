import logging
import os
import numpy as np

from PIL import Image
import cv2

import matplotlib.pyplot as plt

import time

from datasetanalyzerlib.image_similarity.datasets.imagedataset import ImageDataset
from datasetanalyzerlib.enumerations.enums import Extensions
from datasetanalyzerlib.exceptions.exceptions import ExtensionNotFoundException


class ImageLabelDataset(ImageDataset):

    def __init__(self, img_dir: str, label_dir: str, image_files: np.ndarray = None, color_dict: dict=None, background: int=None):
        
        super().__init__(img_dir, image_files)
        self.label_dir = label_dir
        self.color_dict = color_dict
        self.background = background

        logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%H:%M:%S') 

    def _check_label_extensions(self, verbose):
        """
        Checks the extensions of label files in the label directory to ensure consistency.

        Args:
            verbose (bool): If True, logs the process and any issues found.

        Returns:
            str: The extension of the label files if consistent.

        Raises:
            ValueError: If multiple extensions are found in the label directory.
            ExtensionNotFoundException: If the extension is not recognized by the system.
        """
        
        if verbose:
            logging.info(f"Checking label extensions from path: {self.label_dir}...")

        labels_ext = {os.path.splitext(file)[1] for file in os.listdir(self.label_dir)}

        if len(labels_ext) == 1:
            ext = labels_ext.pop()  
            try:
                enum_ext = Extensions.extensionToEnum(ext)

                if enum_ext:
                    print(f"All labels are in {enum_ext.name} format.")
                    return Extensions.enumToExtension(enum_ext)
                
            except ExtensionNotFoundException as e:
                print(f"All labels are in unknown {ext} format.")
                raise e
        else:
            raise ValueError(f"The directory contains multiple extensions for labels: {labels_ext}.")

    def _compare_directories(self, verbose):
        """
        Compares the contents of the image and label directories to check for filename mismatches.

        Args:
            verbose (bool): If True, logs detailed information about discrepancies.

        Raises:
            ValueError: If filenames do not match between the image and label directories.
            FileNotFoundError: If masks or images are missing in either directory.
        """
        if verbose:
            logging.info(f"Comparing directories: {self.img_dir} and {self.label_dir}...")

        images_files = os.listdir(self.img_dir)
        labels_files = os.listdir(self.label_dir)

        images_name = {os.path.splitext(file)[0] for file in os.listdir(self.img_dir)}
        labels_name = {os.path.splitext(file)[0] for file in os.listdir(self.label_dir)}

        if len(images_name) != len(images_files):
            logging.warning(f"Warning: There are duplicate filenames in {self.img_dir}.")

        if len(labels_name) != len(labels_files):
            logging.warning(f"Warning: There are duplicate filenames in {self.label_dir}.")

        if images_name != labels_name:
            differing_files = images_name.symmetric_difference(labels_name)
            if verbose:
                for file in differing_files:
                    logging.warning(f"Filename mismatch: {file} found in one directory but not the other.")
            raise ValueError(f"Filename mismatch between {self.img_dir} and {self.label_dir}. Mismatched files: {differing_files}")
        
        if len(images_name) > len(labels_name):
            missing_masks = images_name - labels_name

            if verbose:
                for missing_mask in missing_masks:
                    logging.warning(f"Image {missing_mask} in {self.img_dir} does not have a mask in {self.label_dir}")
            
            if missing_masks:
                raise FileNotFoundError(f"Missing masks for the following images: {missing_masks}")

        if len(labels_name) > len(images_name):
            missing_images = labels_name - images_name

            if verbose:
                for missing_image in missing_images:
                    logging.warning(f"Mask {missing_image} in {self.label_dir} does not have an image in {self.img_dir}")
            
            if missing_images:
                raise FileNotFoundError(f"Missing images for the following masks: {missing_images}")

        print(f"{self.img_dir} and {self.label_dir} have matching filenames.")
        print(f"Total number of annotated images: {len(images_name)}")

    def _labels_to_array(self, label_files):
        """
        Converts label images into NumPy arrays.

        Args:
            label_files (list): List of paths to the label files.

        Returns:
            list: A list of NumPy arrays representing the labels.
        """
        labels_arr = []

        for label in label_files:
            fpath = os.path.join(self.label_dir, label)
            with Image.open(fpath) as label_img:
                label_arr = np.array(label_img)

                labels_arr.append(label_arr)

        return labels_arr
    
    def _get_classes_from_labels(self, labels, verbose=False):
        """
        Extracts unique classes from the label data.

        Args:
            labels (list): A list of label images represented as NumPy arrays.
            verbose (bool, optional): If True, logs the process and class information. Defaults to False.

        Returns:
            set: A set of unique class identifiers found in the labels.
        """
        if verbose:
            logging.info(f"Checking total number of classes from dataset labels...")
        
        unique_classes = set()
        color_mask = False

        for img_arr in labels:
            if img_arr.ndim == 2:                               #Multilabel or binary label
                unique_classes.update(np.unique(img_arr))
                    
            elif img_arr.ndim == 3 and img_arr.shape[2] == 3:   #RGB Mask
                color_mask=True
                unique_classes.update(map(tuple, img_arr.reshape(-1, 3)))

        if verbose:
            if color_mask:
                print(f"The labels from the dataset are color labels.")
            elif len(unique_classes) == 2:
                print(f"The labels from the dataset are binary labels.")
            else:
                print(f"The labels from the dataset are multiclass.")
        
        print(f"{len(unique_classes)} classes found from dataset labels: {unique_classes}")
        return unique_classes
    
    def _rgb_mask_to_multilabel(self, labels):
        """
        Converts RGB masks into multi-label masks using the provided color dictionary.

        Args:
            labels (list): A list of RGB label images as NumPy arrays.

        Returns:
            list: A list of multi-label masks as NumPy arrays.
        """
        masks = []

        for color_mask in labels:
            multilabel_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)

            pixel_colors = color_mask.reshape(-1, color_mask.shape[-1])

            for idx, pixel_color in enumerate(pixel_colors):
                color_tuple = tuple(pixel_color)
                if color_tuple in self.color_dict:
                    multilabel_mask.reshape(-1)[idx] = self.color_dict[color_tuple]

            multilabel_mask = multilabel_mask.reshape(color_mask.shape[:2])

            masks.append(multilabel_mask)
    
        return masks
    
    def _find_contours(self, labels, verbose):
        """
        Finds contours for objects in the label masks.

        Args:
            labels (list): A list of label masks as NumPy arrays.
            verbose (bool): If True, logs details about the contours found.

        Returns:
            dict: A dictionary where keys are class IDs and values are tuples of
                (list of contours, number of images containing objects of that class).
        """
        contours_dict = {}

        for idx, mask in enumerate(labels):
            unique_classes = np.unique(mask)

            for class_id in unique_classes:
                if self.background is not None and class_id == self.background:
                    continue

                class_mask = np.where(mask == class_id, 255, 0).astype(np.uint8)

                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if class_id not in contours_dict:
                    contours_dict[class_id] = [[], 0]
                contours_dict[class_id][0].extend(contours)  
                contours_dict[class_id][1] += 1  

        contours_dict = {k: v for k, v in sorted(contours_dict.items())}

        if verbose:
            logging.info("Contours for classes:")
            for class_id, (contours, total_count) in contours_dict.items():
                logging.info(f"Class {class_id}: {len(contours)} total objects across {total_count}/{len(labels)} images.")

        return contours_dict
    

    def _compute_metrics(self, contours: dict, plot: bool, output: str):
        """
        Computes metrics about object sizes, bounding boxes, and ellipses for each class.

        Args:
            contours (dict): Dictionary containing contours and image counts for each class.

        Prints:
            Metrics such as object area statistics, bounding box statistics, and ellipse statistics.
        """
        metrics = {"object": [], "bounding_box": [], "ellipse": []}
        class_ids = []

        for class_id, (class_contours, num_images) in contours.items():
            avg_class_objects_per_image = len(class_contours) / num_images

            areas = [cv2.contourArea(contour) for contour in class_contours]

            ellipses_areas = []
            bounding_boxes_areas = []
            for contour in class_contours:
                _,_,w,h = cv2.boundingRect(contour)
                bounding_boxes_areas.append(w * h)
                if len(contour) > 5:
                    ellipse = cv2.fitEllipse(contour)
                    major_axis, minor_axis = ellipse[1]  
                    ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)  
                    ellipses_areas.append(ellipse_area)

            obj_mean = np.mean(areas)
            obj_std = np.std(areas)
            obj_max = max(areas)
            obj_min = min(areas)

            bb_mean = np.mean(bounding_boxes_areas)
            bb_std = np.mean(bounding_boxes_areas)
            bb_max = max(bounding_boxes_areas)
            bb_min = min(bounding_boxes_areas)

            elip_mean = np.mean(ellipses_areas) if ellipses_areas else 0
            elip_std = np.std(ellipses_areas) if ellipses_areas else 0
            elip_max = max(ellipses_areas) if ellipses_areas else 0
            elip_min = min(ellipses_areas) if ellipses_areas else 0

            print(f"------------------------------------")
            print(f"CLASS {class_id} METRICS:")
            print(f"-----------Object metrics-----------")
            print(f"Average objects per image: {avg_class_objects_per_image:.2f}")
            print(f"Average object area: {obj_mean:.2f}")
            print(f"Standard deviation of object area: {obj_std:.2f}")
            print(f"Max object area: {obj_max:.2f}")
            print(f"Min object area: {obj_min:.2f}")
            print(f"-----------Bounding boxes metrics-----------")
            print(f"Average bounding box area: {bb_mean:.2f}")
            print(f"Standard deviation of bounding box area: {bb_std:.2f}")
            print(f"Max bounding box area: {bb_max:.2f}")
            print(f"Min bounding box area: {bb_min:.2f}")
            print(f"-----------Ellipses metrics-----------")
            print(f"Average ellipse area: {elip_mean:.2f}")
            print(f"Standard deviation of ellipse area: {elip_std:.2f}")
            print(f"Max ellipse area: {elip_max:.2f}")
            print(f"Min ellipse area: {elip_min:.2f}")
            print("\n")

            class_ids.append(class_id)
            metrics["object"].append([
                avg_class_objects_per_image, 
                obj_mean, 
                obj_std, 
                obj_max, 
                obj_min
            ])
            metrics["bounding_box"].append([
                bb_mean,
                bb_std,
                bb_max,
                bb_min
            ])
            metrics["ellipse"].append([
                elip_mean,
                elip_std,
                elip_max,
                elip_min
            ])

        if len(class_ids) <= 2:
            print("Metrics won't be plotted since the dataset is not multiclass.")

        if plot:
            metrics_titles = {
                "object": [
                    "Avg. Objects Per Image", 
                    "Avg. Object Area", 
                    "Std. Dev. Object Area", 
                    "Max Object Area", 
                    "Min Object Area"
                ],
                "bounding_box": [
                    "Avg. Bounding Box Area", 
                    "Std. Dev. Bounding Box Area", 
                    "Max Bounding Box Area", 
                    "Min Bounding Box Area"
                ],
                "ellipse": [
                    "Avg. Ellipse Area", 
                    "Std. Dev. Ellipse Area", 
                    "Max Ellipse Area", 
                    "Min Ellipse Area"
                ]
            }
            
            for metric_type, values in metrics.items():
                titles = metrics_titles[metric_type]

                if metric_type == "object":
                    rows, cols = 3, 2
                else:  
                    rows, cols = 2, 2

                _, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
                axs = axs.flatten()  
                class_ids = list(range(len(values)))

                for idx, title in enumerate(titles):
                    axs[idx].bar(class_ids, [v[idx] for v in values])
                    axs[idx].set_title(title)
                    axs[idx].set_xlabel("Class ID")
                    axs[idx].set_ylabel("Value")
                    axs[idx].grid(axis="y")
 
                for idx in range(len(titles), len(axs)):
                    axs[idx].axis("off")

                plt.tight_layout()
                if output:  
                    output_path = os.path.join(output, f"{metric_type}_metrics.png")
                    plt.savefig(output_path, format='png')
                    print(f"Plot saved to {output_path}")
                    plt.close()
                else:
                    plt.show()


    def analyze(self, plot: bool=True, output: str=None, verbose: bool=False):
        """
        Analyzes the dataset of images and corresponding labels to extract metrics and insights.

        Args:
            verbose (bool, optional): If True, logs detailed information about the analysis process. 
                                    Defaults to False.

        Returns:
            None: Outputs various metrics and insights about the dataset to the console.
        """

        if verbose:
            start_time = time.time()


        label_extension = self._check_label_extensions(verbose=verbose)

        self._compare_directories(verbose=verbose)

        label_files = []
        for img_file in self.image_files:
            base_name, _ = os.path.splitext(img_file)
            
            label_file = os.path.join(self.label_dir, f"{base_name}{label_extension}")
            
            if os.path.exists(label_file):  
                label_files.append(label_file)

        print("Image sizes: ")
        self._image_sizes(self.img_dir, self.image_files)
        
        print("Label sizes: ")
        self._image_sizes(self.label_dir, label_files)

        labels_arr = self._labels_to_array(label_files)

        classes_set = self._get_classes_from_labels(labels_arr, verbose)
        
        element = next(iter(classes_set))  
        if isinstance(element, tuple):                 #Color mask (RGB)
            if self.color_dict is None:
                self.color_dict = {v: k for k, v in enumerate(classes_set)}
                
                if verbose:
                    logging.warning(f"Color dictionary for labels is missing, it has been automatically created: {self.color_dict}")

            labels_arr = self._rgb_mask_to_multilabel(labels_arr)
        
        contours_dict= self._find_contours(labels_arr, verbose)
        self._compute_metrics(contours_dict, plot, output)

        if verbose: 
            exection_time = time.time() - start_time
            logging.info(f"Total analysis time: {exection_time: .4f} seconds")
