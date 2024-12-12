import numpy as np
import pydicom
from scipy.ndimage import zoom
from pydicom.filereader import read_dataset
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import os
import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vtk
import plotly.figure_factory as ff
import plotly.io as pio
import shapely
from shapely import Point
from shapely import polygons
from shapely import within
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import statistics
import time




def find_file_with_prefix(folder_path, prefix): #finding the name of RD/RS files from the RT folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith(prefix):
            return os.path.join(folder_path, file_name)
    return None

def read_dicom_rd_file(file_path): #reading and uploding the RD file
    rd = pydicom.dcmread(file_path)
    if rd.Modality == 'RTDOSE':
        return rd
    else:
        raise ValueError("The provided file is not a Radiation Dose (RD) DICOM file.")

RD_file_name = find_file_with_prefix('64147_radiation maps_24082022','RD')
RD_data = read_dicom_rd_file(RD_file_name)


# RD_data.FrameOfReferenceUID : give the name of first CT scan for the dose map
# RD_data.DoseGridScaling : I think that we get the amount of radiation by multiplying the arrays by this parameter
# RD_data.GridFrameOffsetVector : how much the Z position changed for each frame, from the Z position of the first frame (RD_data.FrameOfReferenceUID?)


def show_dose_map(rd_data): #this function opens the dose maps in the form of a heat map.
    grid_scaling = rd_data.DoseGridScaling
    Vmax = (rd_data.pixel_array).max()
    Vmax1 = Vmax*grid_scaling
    for n in range(147,len(rd_data.pixel_array)):
        dose_map = np.flipud(rd_data.pixel_array[n])
        print(dose_map.shape)
        plt.imshow(dose_map*grid_scaling, cmap='jet', origin='lower', vmin=0, vmax=Vmax1)
        plt.colorbar(label='Dose (Gy)')
        plt.title(f'Dose map at slice {n}')
        plt.xlabel('X Axis (pixels)')
        plt.ylabel('Y Axis (pixels)')
        plt.show()


RS_file_name = find_file_with_prefix('64147_radiation maps_24082022','RS')

def read_dicom_rs_file(file_path): #reading and uploding the RS file
    rs = pydicom.dcmread(file_path)
    if rs.Modality == 'RTSTRUCT':
        return rs
    else:
        raise ValueError("The provided file is not an RT Structure Set (RTSTRUCT) DICOM file.")

RS_data = read_dicom_rs_file(RS_file_name)


def get_contour_data(rs_dataset): #Extracting the contour positioning acording to ROI and the collor of each contour
    contour_data = {}
    contour_color = {}
    roi_contour_sequence = rs_dataset.ROIContourSequence
    structure_set_ROI_sequence = rs_dataset.StructureSetROISequence

    for i in range(len(structure_set_ROI_sequence)):
        roi_contour = roi_contour_sequence[i]
        roi_name = structure_set_ROI_sequence[i].ROIName

        contour_data[roi_name] = []
        contour_color[roi_name] = roi_contour.ROIDisplayColor
        contour_sequence = roi_contour.ContourSequence

        for contour_sequence_item in contour_sequence:
            contour_points = contour_sequence_item.ContourData
            contour_data[roi_name] = [contour_points]

    return contour_data, contour_color

(RS_contour_data, Contour_color) = get_contour_data(RS_data)


def load_contour_slices(rs_dataset):
    contour_slices = {}
    roi_contour_sequence = rs_dataset.ROIContourSequence
    structure_set_ROI_sequence = rs_dataset.StructureSetROISequence

    for i in range(len(structure_set_ROI_sequence)):
        roi_contour = roi_contour_sequence[i].ContourSequence
        roi_name = structure_set_ROI_sequence[i].ROIName
        slice_data = []
        contour_slices[roi_name] = []

        for i in range(len(roi_contour)):
            slice_number = roi_contour[i].ContourImageSequence[0].ReferencedSOPInstanceUID
            contour_points = roi_contour[i].ContourData
            slice_data.append((slice_number, contour_points))

        contour_slices[roi_name] = slice_data
    return contour_slices

RS_contour_slices = load_contour_slices(RS_data)

def load_ct_slices(ct_folder_path):
    # Load DICOM CT image series
    ct_slices = {}
    for file_name in os.listdir(ct_folder_path):
        if file_name.startswith('CT'):
            ct_slice = pydicom.dcmread(os.path.join(ct_folder_path, file_name))
            slice_number = ct_slice.SOPInstanceUID
            slice_position = ct_slice.ImagePositionPatient
            pixel_spacing = ct_slice.PixelSpacing

            ct_slices[slice_number] = (ct_slice.pixel_array, slice_position)
            x_spacing = pixel_spacing[0]
            y_spacing = pixel_spacing[1]

    return (ct_slices, x_spacing, y_spacing)

(CT_slices, X_spacing, Y_spacing) = load_ct_slices('64147_SRS_3 METS_24082022')

def re_sort1(ct_slices):
    slices_order = []
    slices_position = [ct_slices[i][1] for i in sorted(ct_slices.keys())]
    for i in range(len(slices_position)):
        slices_order.append(slices_position[i][2])

    slices_order = sorted(slices_order)
    return slices_order

Slices_order1 = re_sort1(CT_slices)

def find_starting_frame(rd_data,ct_folder_path):
    layer_name = rd_data.FrameOfReferenceUID
    start_frame = pydicom.dcmread(os.path.join(ct_folder_path,layer_name))
    return start_frame

Pixel_data = RD_data.PixelData


slice_thicknes = abs(Slices_order1[0]) - abs(Slices_order1[1])
def match_contour_to_ct(contour_slices, ct_slices):
    matched_slices = {}
    for roi_name, contour_slice_list in contour_slices.items():
        matched_slices[roi_name] = []
        for slice_number, contour_points in contour_slice_list:
            if slice_number in ct_slices:
                matched_slices[roi_name].append((slice_number, contour_points, ct_slices[slice_number][0], ct_slices[slice_number][1]))

    return matched_slices

matched_slices = match_contour_to_ct(RS_contour_slices,CT_slices)



def draw_contour_on_dose_map(rd_data, matched_slices, x_spacing, y_spacing, slices_order, contour_color):
    grid_scaling = rd_data.DoseGridScaling
    Vmax = (rd_data.pixel_array).max()
    Vmax1 = Vmax * grid_scaling
    slice_position1 = RD_data.ImagePositionPatient[0]
    slice_position2 = RD_data.ImagePositionPatient[1]
    for n in range(140, len(rd_data.pixel_array)):
        lables = []
        for roi_name, matched_slice_list in matched_slices.items():
            for slice_number, contour_points, ct_image, slice_position in matched_slice_list:
                if slice_position[2] == slices_order[n]:
                    dose_map = rd_data.pixel_array[n]

                    plt.imshow(dose_map * grid_scaling, cmap='cool', origin='lower', vmin=0, vmax=Vmax1)
                    plt.title(f'Dose map at slice {n}')
                    plt.xlabel('X Axis (pixels)')
                    plt.ylabel('Y Axis (pixels)')


                    # Extract x, y coordinates from contour points
                    x = [float(contour_points[i]) for i in range(0, len(contour_points), 3)]
                    y = [float(contour_points[i + 1]) for i in range(0, len(contour_points), 3)]

                    # Convert world coordinates to pixel coordinates
                    # (Assuming ImagePositionPatient gives the position of the first pixel in the image)
                    x_pixel = (np.array(x) - slice_position1) / x_spacing
                    y_pixel = (np.array(y) - slice_position2) / y_spacing
                    normalized_color = [c / 255.0 for c in contour_color[roi_name]]
                    # Plot the contour on the CT image

                    if roi_name in lables:
                        plt.plot(x_pixel, y_pixel, color=normalized_color, linewidth=1)
                    else:
                        plt.plot(x_pixel, y_pixel, color=normalized_color,  label=roi_name, linewidth=1)
                        lables.append(roi_name)
                    plt.legend(loc="upper left", fontsize="9")
                    plt.title(f" Number: {n} out of {len(slices_order)}")

        plt.colorbar(label='Dose (Gy)')
        plt.show()

#draw_contour_on_dose_map(RD_data,matched_slices , RD_data.PixelSpacing[0], RD_data.PixelSpacing[1], Slices_order1, Contour_color)


def draw_dose_map_on_CT(rd_data, matched_slices, slices_order, x_spacing, y_spacing, contour_color):
    grid_scaling = rd_data.DoseGridScaling
    dose_position = RD_data.ImagePositionPatient
    for n in range(len(rd_data.pixel_array)):
        lables = []
        dose_grid = rd_data.pixel_array[n]
        for roi_name, matched_slice_list in matched_slices.items():
            for slice_number, contour_points, ct_image, slice_position in matched_slice_list:
                if slice_position[2] == slices_order[n]:
                    scale_factors = [rd_data.PixelSpacing[0] / x_spacing, rd_data.PixelSpacing[1] / y_spacing]
                    resampled_dose1 = zoom(dose_grid, scale_factors, order=1)
                    Vmax = resampled_dose1.max()
                    Vmax1 = Vmax * grid_scaling


                    y_offset1 = round((dose_position[0] - slice_position[0])/x_spacing)
                    x_offset1 = round((dose_position[1] - slice_position[1])/y_spacing)

                    new_ct = np.zeros_like(resampled_dose1)

                    for x in range(new_ct.shape[0]):
                        for y in range(new_ct.shape[1]):
                            x_index = x + x_offset1
                            y_index = y + y_offset1
                            new_ct[x, y] = ct_image[int(x_index), int(y_index)]



                    plt.imshow(new_ct, cmap='gray')
                    plt.imshow(resampled_dose1*grid_scaling, cmap='cool', origin='lower', vmin=0, vmax=Vmax1, alpha=0.5)

                    x = [float(contour_points[i]) for i in range(0, len(contour_points), 3)]
                    y = [float(contour_points[i + 1]) for i in range(0, len(contour_points), 3)]

                    # Convert world coordinates to pixel coordinates
                    # (Assuming ImagePositionPatient gives the position of the first pixel in the image)
                    x_pixel = ((np.array(x) - slice_position[0]) / x_spacing) - y_offset1
                    y_pixel = (np.array(y) - slice_position[1]) / y_spacing - x_offset1
                    normalized_color = [c / 255.0 for c in contour_color[roi_name]]
                    # Plot the contour on the CT image

                    if roi_name in lables:
                        plt.plot(x_pixel, y_pixel, color=normalized_color, linewidth=1)
                    else:
                        plt.plot(x_pixel, y_pixel, color=normalized_color, label=roi_name, linewidth=1)
                        lables.append(roi_name)
                    plt.legend(loc="upper left", fontsize="9")
                    plt.title(f" Number: {n} out of {len(slices_order)}")
                    plt.title(f'Dose map at slice {n}')
                    plt.xlabel('X Axis (pixels)')
                    plt.ylabel('Y Axis (pixels)')


        plt.colorbar(label='Dose (Gy)')
        plt.show()

#draw_dose_map_on_CT(RD_data, matched_slices, Slices_order1, X_spacing, Y_spacing, Contour_color)



def get_contour_data1(rtstruct):
    """Extract contour data from ROIContourSequence."""
    contour_data = {}
    structure_set_ROI_sequence = rtstruct.StructureSetROISequence
    roi_contour = rtstruct.ROIContourSequence
    for i in range(len(structure_set_ROI_sequence)):
        roi_name = structure_set_ROI_sequence[i].ROIName
        contour_data[roi_name] = []
        roi_contour1 = roi_contour[i]
        for contour_sequence in roi_contour1.ContourSequence:
            contour_points = contour_sequence.ContourData
            if isinstance(contour_points, pydicom.multival.MultiValue):
                points = np.array(contour_points).reshape(-1, 3)  # Reshape into Nx3 array (x, y, z)
            else:
                points = np.array(list(map(float, contour_points.split()))).reshape(-1, 3)
            contour_data[roi_name].append(points)
    return contour_data

RS_contour_data1 = get_contour_data1(RS_data)


#this function do a number of things:
# 1. Create a polygon for each ROI in each layer.
# 2. Calculate for each pixel if it is with in the polygon or not
# 3. Sum the values of all pixels within a given polygon, calculating the amount of radiation in the ROI
def create_histogram(rd_data, matched_slices, slices_order, x_spacing, y_spacing, rs_contour_data, contour_color):
    dose_position = RD_data.ImagePositionPatient
    grid_scaling = rd_data.DoseGridScaling
    histograms = {}
    for n in range(len(rd_data.pixel_array)):
        lables = []
        dose_grid = rd_data.pixel_array[n]
        for roi_name, matched_slice_list in matched_slices.items():
            for slice_number, contour_points, ct_image, slice_position in matched_slice_list:
                if slice_position[2] == slices_order[n]:
                    for roi_name1, contour_data_list in rs_contour_data.items():
                        if roi_name == roi_name1:

                            for contour_data in contour_data_list:


                                if contour_data[0][2] == np.float64("{:.2f}".format(slices_order[n])):
                                    # Outer Contour has been excluded for faster results
                                    if roi_name != 'Outer Contour':


                                        scale_factors = [rd_data.PixelSpacing[0] / x_spacing, rd_data.PixelSpacing[1] / y_spacing]
                                        y_offset1 = round((dose_position[0] - slice_position[0]) / x_spacing)
                                        x_offset1 = round((dose_position[1] - slice_position[1]) / y_spacing)
                                        #print(scale_factors)
                                        resampled_dose1 = zoom(dose_grid, scale_factors, order=1)
                                        resampled_dose2 = resampled_dose1
                                        Vmax = resampled_dose1.max()
                                        Vmax1 = Vmax * grid_scaling



                                        points = []
                                        #for i in range(len(contour_data)):
                                        #    contour_data_layer = contour_data[i]
                                        for j in range(contour_data.shape[0]):
                                            points.append((contour_data[j][0] - dose_position[0]) / x_spacing)
                                            points.append((contour_data[j][1] - dose_position[1]) / y_spacing)

                                        points = np.array(points).reshape(-1, 2)
                                        polygon = Polygon(points)
                                        #print(len(points))

                                        radiation = []
                                        for x in range(resampled_dose1.shape[1]):
                                            for y in range(resampled_dose1.shape[0]):

                                                if within(Point(x, y), polygon):
                                                    radiation.append(resampled_dose1[y, x] * grid_scaling)
                                                    resampled_dose2[y, x] = Vmax



                                        plt.imshow(resampled_dose2 * grid_scaling, cmap='cool', origin='lower', vmin=0, vmax=Vmax1, alpha=0.5)


                                        if roi_name in histograms.keys():
                                            radiation1 = histograms[roi_name] + radiation
                                            histograms[roi_name] = radiation1
                                            roi_area = histograms[roi_name + ' Area'] + polygon.area
                                            histograms[roi_name + ' Area'] = roi_area
                                        else:
                                            histograms[roi_name] = radiation
                                            histograms[roi_name + ' Area'] = polygon.area

                                        x = [float(contour_points[i]) for i in range(0, len(contour_points), 3)]
                                        y = [float(contour_points[i + 1]) for i in range(0, len(contour_points), 3)]

                                        # Convert world coordinates to pixel coordinates
                                        # (Assuming ImagePositionPatient gives the position of the first pixel in the image)
                                        x_pixel = ((np.array(x) - slice_position[0]) / x_spacing) - y_offset1
                                        y_pixel = (np.array(y) - slice_position[1]) / y_spacing - x_offset1
                                        normalized_color = [c / 255.0 for c in contour_color[roi_name]]
                                        # Plot the contour on the CT image

#Draw contours of each ROI in addition to the polygon
                #                         if roi_name in lables:
                #                             plt.plot(x_pixel, y_pixel, color=normalized_color, linewidth=1)
                #                             plt.plot(*polygon.exterior.xy)
                #                         else:
                #                             plt.plot(x_pixel, y_pixel, color=normalized_color, label=roi_name, linewidth=1)
                #                             plt.plot(*polygon.exterior.xy)
                #                             lables.append(roi_name)
                #
                #
                #                         plt.legend(loc="upper left", fontsize="9")
                #                         plt.title(f" Number: {n} out of {len(slices_order)}")
                #                         plt.title(f'Dose map at slice {n}')
                #                         plt.xlabel('X Axis (pixels)')
                #                         plt.ylabel('Y Axis (pixels)')
                #
                #
                # plt.show()

    return histograms

start_time = time.time()
histograms = create_histogram(RD_data, matched_slices, Slices_order1, X_spacing, Y_spacing, RS_contour_data1, Contour_color)
print("--- %s seconds ---" % (time.time() - start_time))

#creating and displaying the histogram of each ROI including key parameters
for roi in RS_contour_data1.keys():
    # Outer Contour has been excluded for faster results
    if roi != 'Outer Contour':

        f, ax = plt.subplots()
        plt.hist(histograms[roi], bins=30, color='skyblue', edgecolor='black')
        # Adding labels and title
        plt.xlabel('Gray')
        plt.ylabel('Number of Voxels')
        plt.title(roi)

        plt.text(.51, .99, f"Min = {min(histograms[roi])}", ha='left', va='top', transform=ax.transAxes)
        plt.text(.51, .95, f"Max = {max(histograms[roi])}", ha='left', va='top', transform=ax.transAxes)
        plt.text(.51, .91, f"Mean = {statistics.mean(histograms[roi])}", ha='left', va='top', transform=ax.transAxes)
        plt.text(.51, .87, f"Median = {statistics.median(histograms[roi])}", ha='left', va='top', transform=ax.transAxes)
        #This line of code give the wrong results, probebly the Area is not calculated corectly 
        #plt.text(.51, .83, f"Volum = {(histograms[roi + ' Area']) * slice_thicknes}", ha='left', va='top', transform=ax.transAxes)

    # Display the plot
        plt.show()












