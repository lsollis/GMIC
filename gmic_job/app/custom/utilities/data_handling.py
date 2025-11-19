# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""
Defines utility functions for managing the dataset.
"""
from constants.constants import VIEWS


def unpack_exam_into_images(exam_list, cropped=False, skip_missing=True):
    """
    Turn exam_list into image_list, optionally skipping images with missing metadata
    """
    image_list = []
    skipped_count = 0
    
    for i, exam in enumerate(exam_list):
        for view in VIEWS.LIST:
            for j, image in enumerate(exam[view]):
                image_dict = dict(
                    short_file_path=image,
                    horizontal_flip=exam['horizontal_flip'],
                    full_view=view,
                    side=view[0],
                    view=view[2:],
                )
                
                if cropped:
                    # Check if this image has the required metadata
                    has_metadata = (
                        'window_location' in exam and 
                        view in exam['window_location'] and 
                        j < len(exam['window_location'][view])
                    )
                    
                    if has_metadata:
                        image_dict["window_location"] = exam['window_location'][view][j]
                        image_dict["rightmost_points"] = exam['rightmost_points'][view][j]
                        image_dict["bottommost_points"] = exam['bottommost_points'][view][j]
                        image_dict["distance_from_starting_side"] = exam['distance_from_starting_side'][view][j]
                        image_list.append(image_dict)
                    elif skip_missing:
                        skipped_count += 1
                        print(f"Skipping {image} - missing cropping metadata")
                    else:
                        raise KeyError(f"Missing metadata for {image}")
                else:
                    image_list.append(image_dict)
    
    if cropped and skipped_count > 0:
        print(f"Skipped {skipped_count} images due to missing cropping metadata")
    
    return image_list


def add_metadata(exam_list, additional_metadata_name, additional_metadata_dict):
    """
    Includes new information about images into exam_list
    """
    for exam in exam_list:
        assert additional_metadata_name not in exam, "this metadata is already included"
        exam[additional_metadata_name] = dict()
        for view in VIEWS.LIST:
            exam[additional_metadata_name][view] = []
            for j, image in enumerate(exam[view]):
                exam[additional_metadata_name][view].append(additional_metadata_dict[image])