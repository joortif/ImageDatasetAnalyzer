import logging
import os
from typing import List
import numpy as np
import time

import pandas as pd
from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt

from imagedatasetanalyzer.datasets.imagedataset import ImageDataset
from imagedatasetanalyzer.utils.preprocessing import preprocess_mask

class ImageLabelDataset(ImageDataset):
    """
    Represents a labeled image dataset, extending the functionality of ImageDataset.

    Attributes:
        img_dir (str): Directory containing image files.
        label_dir (str): Directory containing label files.
        image_files (np.ndarray): Array of image filenames (optional).
        color_dict (dict): Mapping between RGB values and class labels (optional).
        background (int): Identifier for the background class (optional).
    """

    def __init__(self, img_dir: str, label_dir: str, image_files: np.ndarray = None, color_dict: dict=None, class_map: dict=None, background: int=None):
        """
        Args:
            img_dir (str): Directory containing image files.
            label_dir (str): Directory containing label files.
            output_dir (str, optional): Directory to save the conversions of labels to multilabel. This transformation is done when labels are in JSON, TXT format or their type is multicolor. 
            image_files (np.ndarray, optional): Array of image filenames to load. If None, all images from the directory are loaded.
            color_dict (dict, optional): Mapping between RGB values and class labels. If None, it is assumed that labels are already in a format that can be mapped to integers.
            class_map (dict, optional): Mapping between class ids and class names.
            background (int, optional): Identifier for the background class. If None, no background class is considered.
        """
        super().__init__(img_dir, image_files)
        self.label_dir = label_dir
        self.color_dict = color_dict
        self.background = background
        self.class_map = class_map

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        self.log_file = None

    def compare_directories(self, verbose):
        """
        Compares the contents of the image and label directories to check for filename mismatches.

        Args:
            verbose (bool): If True, logs detailed information about discrepancies.

        Raises:
            ValueError: If filenames do not match between the image and label directories.
            FileNotFoundError: If masks or images are missing in either directory.
        """
        if verbose:
            self.logger.info(f"Comparing directories: {self.img_dir} and {self.label_dir}...")

        images_files = os.listdir(self.img_dir)
        labels_files = os.listdir(self.label_dir)

        image_names = {os.path.splitext(file)[0] for file in os.listdir(self.img_dir)}
        label_names = {os.path.splitext(file)[0] for file in os.listdir(self.label_dir)}

        if len(image_names) != len(images_files):
            self.logger.warning(f"Warning: There are duplicate filenames in {self.img_dir}.")

        if len(label_names) != len(labels_files):
            self.logger.warning(f"Warning: There are duplicate filenames in {self.label_dir}.")

        missing_masks = image_names - label_names
        missing_images = label_names - image_names

        if missing_masks:
            if verbose:
                for name in missing_masks:
                    self.logger.warning(f"Image '{name}' in {self.img_dir} does not have a corresponding mask in {self.label_dir}")
            raise FileNotFoundError(f"Missing masks for the following images: {missing_masks}")

        if missing_images:
            if verbose:
                for name in missing_images:
                    self.logger.warning(f"Mask '{name}' in {self.label_dir} does not have a corresponding image in {self.img_dir}")
            raise FileNotFoundError(f"Missing images for the following masks: {missing_images}")

        self.logger.info(f"{self.img_dir} and {self.label_dir} have matching filenames.")
        self.logger.info(f"Total number of annotated images: {len(image_names)}")

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

            img_color = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if len(img_color.shape) == 2:
                label_arr = img_color  
            elif len(img_color.shape) == 3 and img_color.shape[2] == 3:
                b, g, r = cv2.split(img_color)
                if np.array_equal(b, g) and np.array_equal(b, r):
                    label_arr = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                else:
                    label_arr = img_color  
            else:
                label_arr = img_color

            labels_arr.append(label_arr)

        return labels_arr
    
    def get_classes_from_labels(self, labels, verbose=False):
        """
        Extracts unique classes from the label data.

        Args:
            labels (list): A list of label images represented as NumPy arrays.
            verbose (bool, optional): If True, logs the process and class information. Defaults to False.

        Returns:
            set: A set of unique class identifiers found in the labels.
        """
        if verbose:
            self.logger.info(f"Checking total number of classes from dataset labels...")
        
        unique_classes = set()

        for img_arr in tqdm(labels, desc="Reading labels"):
            if img_arr.ndim == 2:                               #Multilabel or binary label
                unique_classes.update(np.unique(img_arr))
                    
            elif img_arr.ndim == 3 and img_arr.shape[2] == 3:   #RGB Mask
                unique_classes.update(map(tuple, img_arr.reshape(-1, 3)))

        if verbose:
            if len(unique_classes) == 2:
                self.logger.info("The labels from the dataset are binary.")
            else:
                self.logger.info("The labels from the dataset are multiclass.")
        
        self.logger.info("%d classes found from dataset labels: %s", len(unique_classes), unique_classes)
        return unique_classes
    
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

        for _, mask in enumerate(labels):
            unique_classes = np.unique(mask)

            for class_id in unique_classes:
                if self.background is not None and class_id == self.background:
                    continue

                class_mask = np.where(mask == class_id, 255, 0).astype(np.uint8)

                clean_mask = preprocess_mask(class_mask)

                contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

                if class_id not in contours_dict:
                    contours_dict[class_id] = [[], 0]
                    
                contours_dict[class_id][0].extend(contours)  
                contours_dict[class_id][1] += 1  

        contours_dict = {k: v for k, v in sorted(contours_dict.items())}

        if verbose:
            self.logger.info("Contours for classes:")
            for class_id, (contours, total_count) in contours_dict.items():
                class_name = self.class_map[class_id] if self.class_map is not None and class_id in self.class_map else class_id
                self.logger.info("Class %s: %d total objects across %d/%d images.", class_name, len(contours), total_count, len(labels))

        return contours_dict
    
    def _show_boxplot(self, object_areas: dict, output: str=None):
        num_classes = len(object_areas)
        cols = min(4, num_classes)  
        rows = (num_classes + cols - 1) // cols  

        _, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
        axs = axs.flatten()  

        for idx, (class_id, areas) in enumerate(object_areas.items()):
            class_name = self.class_map[class_id] if self.class_map is not None and class_id in self.class_map else class_id
            axs[idx].boxplot(areas, labels=[f"Class {class_name}"])
            axs[idx].set_title(f"Class {class_name} Area Distribution")
            axs[idx].grid(axis="y")

        for idx in range(len(object_areas), len(axs)):
            axs[idx].axis("off")

        plt.tight_layout()

        if output:
            boxplot_path = os.path.join(output, "object_areas_boxplot.png")
            plt.savefig(boxplot_path, format='png')
            self.logger.info("Boxplot saved to %s", boxplot_path)
            plt.close()
    
    def _save_metrics_csv(self, metrics, output):
        columns = [
            "Class name",
            "Total objects",
            "Num. images with objects",
            "Avg. objects per image", 
            "Avg. object area", 
            "Std. Dev. object area", 
            "Max object area", 
            "Min object area",
            "Avg. bounding box area", 
            "Std. Dev. bounding box area", 
            "Max bounding box area", 
            "Min bounding box area",
            "Avg. ellipse area", 
            "Std. Dev. ellipse area", 
            "Max ellipse area", 
            "Min ellipse area"
        ]

        df = pd.DataFrame(metrics, columns=columns)

        output_path = os.path.join(output, "metrics.csv")

        df.to_csv(output_path, index=False, sep=";", decimal=",")
        self.logger.info("Metrics saved to CSV at %s", output_path)
        return


    def _compute_metrics(self, contours: dict, plot: bool=True, output: str=None):
        """
        Computes metrics about object sizes, bounding boxes, and ellipses for each class.

        Args:
            contours (dict): Dictionary containing contours and image counts for each class.

        Prints:
            Metrics such as object area statistics, bounding box statistics, and ellipse statistics.
        """
        metrics = {"object": [], "bounding_box": [], "ellipse": []}
        class_ids = []
        object_areas_by_class = {}

        csv_data = []

        for class_id, (class_contours, num_images) in contours.items():
            total_objects = len(class_contours)
            avg_class_objects_per_image = len(class_contours) / num_images

            areas = [cv2.contourArea(contour) for contour in class_contours]
            object_areas_by_class[class_id] = areas

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

            obj_mean = np.mean(areas) if areas else 0
            obj_std = np.std(areas) if areas else 0
            obj_max = max(areas) if areas else 0
            obj_min = min(areas) if areas else 0

            bb_mean = np.mean(bounding_boxes_areas) if bounding_boxes_areas else 0
            bb_std = np.std(bounding_boxes_areas) if bounding_boxes_areas else 0
            bb_max = max(bounding_boxes_areas) if bounding_boxes_areas else 0
            bb_min = min(bounding_boxes_areas) if bounding_boxes_areas else 0

            elip_mean = np.mean(ellipses_areas) if ellipses_areas else 0
            elip_std = np.std(ellipses_areas) if ellipses_areas else 0
            elip_max = max(ellipses_areas) if ellipses_areas else 0
            elip_min = min(ellipses_areas) if ellipses_areas else 0

            class_name = self.class_map[class_id] if self.class_map is not None and class_id in self.class_map else class_id

            self.logger.info("------------------------------------")
            self.logger.info("CLASS %s METRICS:", str(class_name).upper())
            self.logger.info("-----------Object metrics-----------")
            self.logger.info("Average objects per image: %.2f", avg_class_objects_per_image)
            self.logger.info("Average object area: %.2f", obj_mean)
            self.logger.info("Standard deviation of object area: %.2f", obj_std)
            self.logger.info("Max object area: %.2f", obj_max)
            self.logger.info("Min object area: %.2f", obj_min)
            self.logger.info("-----------Bounding boxes metrics-----------")
            self.logger.info("Average bounding box area: %.2f", bb_mean)
            self.logger.info("Standard deviation of bounding box area: %.2f", bb_std)
            self.logger.info("Max bounding box area: %.2f", bb_max)
            self.logger.info("Min bounding box area: %.2f", bb_min)
            self.logger.info("-----------Ellipses metrics-----------")
            self.logger.info("Average ellipse area: %.2f", elip_mean)
            self.logger.info("Standard deviation of ellipse area: %.2f", elip_std)
            self.logger.info("Max ellipse area: %.2f", elip_max)
            self.logger.info("Min ellipse area: %.2f", elip_min)
            self.logger.info("\n")

            csv_data.append([
                class_name, 
                total_objects,
                num_images,
                f"{avg_class_objects_per_image:.2f}".replace('.', ','), 
                f"{obj_mean:.2f}".replace('.', ','), 
                f"{obj_std:.2f}".replace('.', ','), 
                f"{obj_max:.2f}".replace('.', ','), 
                f"{obj_min:.2f}".replace('.', ','),

                f"{bb_mean:.2f}".replace('.', ','), 
                f"{bb_std:.2f}".replace('.', ','), 
                f"{bb_max:.2f}".replace('.', ','), 
                f"{bb_min:.2f}".replace('.', ','),

                f"{elip_mean:.2f}".replace('.', ','), 
                f"{elip_std:.2f}".replace('.', ','), 
                f"{elip_max:.2f}".replace('.', ','), 
                f"{elip_min:.2f}".replace('.', ',')
            ])

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

        self._save_metrics_csv(csv_data, output)

        if len(class_ids) <= 1:
            self.logger.info("Metrics won't be plotted since the dataset has only one class.")
            return

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

            self._show_boxplot(object_areas_by_class, output)
            
            for metric_type, values in metrics.items():
                titles = metrics_titles[metric_type]

                if metric_type == "object":
                    rows, cols = 3, 2
                else:  
                    rows, cols = 2, 2

                _, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
                axs = axs.flatten()  
                class_indices = range(len(class_ids)) 

                class_labels = [self.class_map[cid] if self.class_map is not None and cid in self.class_map else str(cid) for cid in class_ids]

                only_numbers = all(label.isdigit() for label in class_labels)

                for idx, title in enumerate(titles):
                    axs[idx].bar(class_indices, [v[idx] for v in values])
                    axs[idx].set_title(title)
                    axs[idx].grid(axis="y")
                    axs[idx].set_xticks(class_indices)  
                    axs[idx].set_xticklabels(class_labels, rotation=45 if not only_numbers else 0)
 
                for idx in range(len(titles), len(axs)):
                    axs[idx].axis("off")

                plt.tight_layout()
                if output:  
                    output_path = os.path.join(output, f"{metric_type}_metrics.png")
                    plt.savefig(output_path, format='png')
                    self.logger.info("Plot saved to %s", output_path)
                    plt.close()


    def analyze(self, plot: bool=True, similarity_index: List[str]=["SSIM", "LPIPS"], output: str=None, verbose: bool=False):
        """
        Analyzes the dataset of images and corresponding labels to extract metrics and insights.

        Args:
            verbose (bool, optional): If True, logs detailed information about the analysis process. 
                                    Defaults to False.

        Returns:
            None: Outputs various metrics and insights about the dataset to the console.
        """

        if output is None:
            output = os.getcwd()
        os.makedirs(output, exist_ok=True)

        log_path = os.path.join(output, "logs.txt")
        self.log_file = log_path

        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(file_handler)

        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self.logger.addHandler(console_handler)

        if verbose:
            start_time = time.time()
            self.logger.info("Starting dataset analysis. Saving log to %s", log_path)

        self.compare_directories(verbose=verbose)

        label_files = []
        for img_file in self.image_files:
            base_name, _ = os.path.splitext(img_file)
            
            label_file = os.path.join(self.label_dir, f"{base_name}.png")
            
            if os.path.exists(label_file):  
                label_files.append(f"{base_name}.png")
            
        

        self.dataset_similarity(similarity_index, self.logger)

        labels_arr = self._labels_to_array(label_files)

        classes = self.get_classes_from_labels(labels_arr, verbose)

        if self.class_map is not None and len(self.class_map) != 0 and len(self.class_map) != len(classes):
            self.logger.info(
                "Warning: %d classes found automatically but class map contains %d entries. Provided class map won't be used.",
                len(classes),
                len(self.class_map)
            )
            self.class_map = {}
        
        contours_dict= self._find_contours(labels_arr, verbose)
        self._compute_metrics(contours_dict, plot, output)

        if verbose: 
            exection_time = time.time() - start_time
            self.logger.info("Total analysis time: %.4f seconds", exection_time)
