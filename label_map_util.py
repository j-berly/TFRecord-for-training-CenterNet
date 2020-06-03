# https://github.com/google/automl/blob/master/efficientdet/dataset/label_map_util.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def create_category_index(categories):
    """
    Creates dictionary of COCO compatible categories keyed by category id.

    Args:
      categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
          e.g., 'cat', 'dog', 'pizza'.

    Returns:
      category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index
