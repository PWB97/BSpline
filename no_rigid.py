import os
import SimpleITK as sitk
import numpy as np
import cv2

dimension = 2
spline_order = 3
direction_matrix_row_major = [1., 0., 0., 1.] # identity, mesh is axis aligned
domain_physical_dimension = [232, 320]
np.random.seed(1)

IMG_PATH = '/mnt/hd1/puwenbo/Dataset/registration2D_dataset/image/CT'
SAVE_PATH = '/mnt/hd1/puwenbo/Dataset/registration2D_dataset/image/bs_moving'
TRANS_PATH = '/mnt/hd1/puwenbo/Dataset/registration2D_dataset/image/bs_trans'

def bspline_trans(input, save_name):
    origin = input.GetOrigin()
    bspline = sitk.BSplineTransform(dimension, spline_order)
    bspline.SetTransformDomainOrigin(origin)
    bspline.SetTransformDomainDirection(direction_matrix_row_major)
    bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimension)
    bspline.SetTransformDomainMeshSize((25, 25))

    originalControlPointDisplacements = np.random.random(len(bspline.GetParameters())) * 8
    np.savetxt(os.path.join(TRANS_PATH, save_name) + '.txt', originalControlPointDisplacements)
    bspline.SetParameters(originalControlPointDisplacements)

    transformed_img = sitk.Resample(input, bspline)
    sitk.WriteImage(transformed_img, os.path.join(SAVE_PATH, save_name))


def do_trans(PATH):
    for file in os.listdir(PATH):
        file_path = os.path.join(PATH, file)
        assert os.path.exists(file_path), 'FILE WRONG'
        moving = np.array(cv2.imread(file_path, 0))
        moving_img = sitk.GetImageFromArray(moving)
        bspline_trans(moving_img, file)


if __name__ == '__main__':
    os.mkdir(SAVE_PATH)
    os.mkdir(TRANS_PATH)
    do_trans(IMG_PATH)