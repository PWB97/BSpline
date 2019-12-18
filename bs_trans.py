import SimpleITK as sitk
import numpy as np
import cv2
import neuron.plot as plot
import os

np.random.seed(1)

moving = np.array(cv2.imread('moving.bmp', 0))
print(moving.shape())

moving_img = sitk.GetImageFromArray(moving)


dimension = 2
spline_order = 3
direction_matrix_row_major = [1., 0., 0., 1.] # identity, mesh is axis aligned
origin = moving_img.GetOrigin()
domain_physical_dimension = [232, 320]

bspline = sitk.BSplineTransform(dimension, spline_order)
bspline.SetTransformDomainOrigin(origin)
bspline.SetTransformDomainDirection(direction_matrix_row_major)
bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimension)
bspline.SetTransformDomainMeshSize((25, 25))


originalControlPointDisplacements = np.random.random(len(bspline.GetParameters()))*8
print(originalControlPointDisplacements)
print(np.shape(originalControlPointDisplacements))
bspline.SetParameters(originalControlPointDisplacements)

# print(originalControlPointDisplacements)
# print(bspline.GetParameters())
# print(min(originalControlPointDisplacements), max(originalControlPointDisplacements))
# print(bspline)
# print(np.shape(bspline.GetParameters()))

transformed_img = sitk.Resample(moving_img, bspline)

transform_to_displacement_filter = sitk.TransformToDisplacementFieldFilter()
transform_to_displacement_filter.SetReferenceImage(moving_img)
displacement_transform = sitk.DisplacementFieldTransform(transform_to_displacement_filter.Execute(bspline))
displacement_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)
# print(displacement_transform)
moved_grid = sitk.Resample(moving_img, displacement_transform)

field = displacement_transform.GetDisplacementField()

flow = np.array(field).reshape(232, 320, 2)

flow = np.expand_dims(flow, axis=0)

plot.flow(flow[:,0:28,0:28,:], width=5)



# print(transformed_img)



# sitk.WriteImage(transformed_img, 'test.bmp')