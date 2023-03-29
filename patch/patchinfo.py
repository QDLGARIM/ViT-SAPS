# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 09:48:59 2022

@author: eyxysdht
"""


import numpy as np

class PatchInfo():
    """Information of a certain patch in an image."""
    
    def __init__(self, patch, img_size, patch_size, row, col):
        self._patch = patch
        self._imgSize = img_size      # Size of the original image
        self._patchSize = patch_size   # Size of the current patch
        self._row = row    # The row number of the patch's top left corner pixel in the original image
        self._col = col    # The column number of the patch's top left corner pixel in the original image


class CenterPoint():
    """Information of the center point of a certain patch in an image.
    Used in uneven interpolation."""
    
    def __init__(self, row_topleft, col_topleft, size, formula, isPatch, ori_index=None):
        if isPatch == True and ori_index == None:
            raise ValueError("For patch points, its index in the original patch distribution data structure should be provided.")
        self._y = row_topleft + size/2
        self._x = col_topleft + size/2
        self._size = size
        self._isPatchPoint = isPatch   # Whether it is a center point of a real patch or a interpolated point
        self._oriIndex = ori_index
        self._formula = formula
        self._isAreaPoint = [False, False, False, False, False, False, False, False, False]
    
    def coord(self):
        return (self._y, self._x)
    
    def size(self):
        return self._size
    
    def isPatchPoint(self):
        return self._isPatchPoint
    
    def oriIndex(self):
        return self._oriIndex
    
    def formula(self):
        return self._formula
    
    def setAreaPoint(self, index):
        self._isAreaPoint[index] = True
    
    def clearAreaPoint(self, index):
        self._isAreaPoint[index] = False
    
    def hasInnerArea(self):
        return self._isAreaPoint[0]    # Whether it is the top left corner point of a square inner interpolation area
    
    def hasTopArea(self):
        return self._isAreaPoint[1]     # Whether it is the left corner point of a top edge interpolation area
    
    def hasBottomArea(self):
        return self._isAreaPoint[2]     # Whether it is the left corner point of a bottom edge interpolation area
    
    def hasLeftArea(self):
        return self._isAreaPoint[3]     # Whether it is the top corner point of a left edge interpolation area
    
    def hasRightArea(self):
        return self._isAreaPoint[4]     # Whether it is the top corner point of a right edge interpolation area
    
    def hasTopLeftArea(self):
        return self._isAreaPoint[5]    # Whether it is the corner point of a top left corner interpolation area
    
    def hasTopRightArea(self):
        return self._isAreaPoint[6]   # Whether it is the corner point of a top right corner interpolation area
    
    def hasBottomLeftArea(self):
        return self._isAreaPoint[7]   # Whether it is the corner point of a bottom left corner interpolation area
    
    def hasBottomRightArea(self):
        return self._isAreaPoint[8]   # Whether it is the corner point of a bottom right corner interpolation area
        
        
class CenterPointSet():
    """A set of center points of patches which have the same size and in a same image."""
    
    def __init__(self, size):
        self._items = list()
        self._coords = list()
        self._size = size
    
    def __len__(self):
        """The number of elements in this set."""
        return len(self._items)
    
    def size(self):
        return self._size
    
    def reset(self):
        self._items = list()
        self._coords = list()
    
    def add(self, item):
        if item.size() != self.size():
            raise ValueError("Patch size of the new item must be equal to that of this set.")
        self._items.append(item)
        self._coords.append(item.coord())
    
    def __getitem__(self, index):
        """Subscript operator for access at index."""
        return self._items[index]

    def __setitem__(self, index, newItem):
        """Subscript operator for replacement at index."""
        self._items[index] = newItem
    
    def __iter__(self):
        """Supports iteration over a view of this set."""
        return iter(self._items)
    
    def setAreaPoint(self, target_size):
        for item in self._items:
            if (item.coord()[0], item.coord()[1]+self.size()) in self._coords and (item.coord()[0]+self.size(), item.coord()[1]) in self._coords and (item.coord()[0]+self.size(), item.coord()[1]+self.size()) in self._coords:
                item.setAreaPoint(0)    # This point is the top left corner point of an inner interpolation area.
                
            flags = [False, False, False, False]
            if item.coord()[0] - self.size()//2 == 0:
                flags[0] = True     # This patch is at the top edge
            if item.coord()[0] + self.size()//2 == target_size[0]:
                flags[1] = True     # This patch is at the bottom edge
            if item.coord()[1] - self.size()//2 == 0:
                flags[2] = True     # This patch is at the left edge
            if item.coord()[1] + self.size()//2 == target_size[1]:
                flags[3] = True     # This patch is at the right edge
                
            if flags[0] and (item.coord()[0], item.coord()[1]+self.size()) in self._coords:
                item.setAreaPoint(1)
            if flags[1] and (item.coord()[0], item.coord()[1]+self.size()) in self._coords:
                item.setAreaPoint(2)
            if flags[2] and (item.coord()[0]+self.size(), item.coord()[1]) in self._coords:
                item.setAreaPoint(3)
            if flags[3] and (item.coord()[0]+self.size(), item.coord()[1]) in self._coords:
                item.setAreaPoint(4)
            if flags[0] and flags[2]:
                item.setAreaPoint(5)
            if flags[0] and flags[3]:
                item.setAreaPoint(6)
            if flags[1] and flags[2]:
                item.setAreaPoint(7)
            if flags[1] and flags[3]:
                item.setAreaPoint(8)
    
    def findPoint(self, targetCoord):
        try:
            index = self._coords.index(targetCoord)
            return self._items[index] 
        except ValueError:
            return None
    