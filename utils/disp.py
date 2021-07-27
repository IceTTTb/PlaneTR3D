import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os


def plot_depth_recall_curve(method_recalls, type='', save_path=None, method_color=None):
    assert type in ['pixel', 'PIXEL', 'Pixel', 'plane', 'PLANE', 'Plane']
    depth_threshold = np.arange(0, 0.65, 0.05)
    title = 'Per-'+type+' Recall(%)'

    pre_defined_recalls = {}
    if type in ['pixel', 'PIXEL', 'Pixel']:
        recall_planeAE = np.array(
            [0., 30.59, 51.88, 62.83, 68.54, 72.13, 74.28, 75.38, 76.57, 77.08, 77.35, 77.54, 77.86])
        pre_defined_recalls['PlaneAE'] = recall_planeAE

        recall_planeNet = np.array(
            [0., 22.79, 42.19, 52.71, 58.92, 62.29, 64.31, 65.20, 66.10, 66.71, 66.96, 67.11, 67.14])
        pre_defined_recalls['PlaneNet'] = recall_planeNet

    else:
        recall_planeAE = np.array(
            [0., 22.93, 40.17, 49.40, 54.58, 57.75, 59.72, 60.92, 61.84, 62.23, 62.56, 62.76, 62.93])
        pre_defined_recalls['PlaneAE'] = recall_planeAE

        recall_planeNet = np.array(
            [0., 15.78, 29.15, 37.48, 42.34, 45.09, 46.91, 47.77, 48.54, 49.02, 49.33, 49.53, 49.59])
        pre_defined_recalls['PlaneNet'] = recall_planeNet

    plt.figure(figsize=(5, 4))
    plt.xlabel('Depth Threshold', fontsize=18)
    plt.ylabel(title, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    markers = {'PlaneNet': 'o', 'PlaneAE': '*'}
    colors = {'PlaneNet': 'gray', 'PlaneAE': '#FFCC99'}
    for method_name, recalls in pre_defined_recalls.items():
        assert len(depth_threshold) == len(recalls)
        plt.plot(depth_threshold, recalls, linewidth=3, marker=markers[method_name],label=method_name, color=colors[method_name])

    for method_name, recalls in method_recalls.items():
        assert len(depth_threshold) == len(recalls)
        if method_color is not None:
            plt.plot(depth_threshold, recalls, linewidth=3, marker='^', color=method_color[method_name], label=method_name)
        else:
            plt.plot(depth_threshold, recalls, linewidth=3, marker='^', label=method_name, color='#FF6666')

    plt.legend(loc='lower right', fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    x_major_locator = MultipleLocator(0.1)
    y_major_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'depth_recall_%s.png'%(type)))
    else:
        plt.savefig('../results/depth_recall_%s.png'%(type))
    plt.close()


def plot_normal_recall_curve(method_recalls, type='', save_path=None, method_color=None):
    assert type in ['pixel', 'PIXEL', 'Pixel', 'plane', 'PLANE', 'Plane']
    normal_threshold = np.linspace(0.0, 30, 13)
    title = 'Per-'+type+' Recall(%)'

    pre_defined_recalls = {}
    if type in ['pixel', 'PIXEL', 'Pixel']:
        recall_planeAE = np.array(
            [0., 30.20, 59.89, 69.79, 73.59, 75.67, 76.8, 77.3, 77.42, 77.57, 77.76, 77.85, 78.03])
        pre_defined_recalls['PlaneAE'] = recall_planeAE

        recall_planeNet = np.array(
            [0., 19.68, 43.78, 57.55, 63.36, 65.27, 66.03, 66.64, 66.99, 67.16, 67.20, 67.26, 67.29])
        pre_defined_recalls['PlaneNet'] = recall_planeNet
    else:
        recall_planeAE = np.array(
            [0., 20.05, 42.66, 51.85, 55.92, 58.34, 59.52, 60.35, 60.75, 61.23, 61.64, 61.84, 61.93])
        pre_defined_recalls['PlaneAE'] = recall_planeAE

        recall_planeNet = np.array(
            [0., 12.49, 29.70, 40.21, 44.92, 46.77, 47.71, 48.44, 48.83, 49.09, 49.20, 49.31, 49.38])
        pre_defined_recalls['PlaneNet'] = recall_planeNet

    plt.figure(figsize=(5, 4))
    plt.xlabel('Normal Threshold', fontsize=18)
    plt.ylabel(title, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    markers = {'PlaneNet': 'o', 'PlaneAE': '*', 'PlaneRCNN': '.'}
    colors = {'PlaneNet': 'gray', 'PlaneAE': '#FFCC99', 'PlaneRCNN': 'mediumaquamarine'}
    for method_name, recalls in pre_defined_recalls.items():
        assert len(normal_threshold) == len(recalls)
        plt.plot(normal_threshold, recalls, linewidth=3, marker=markers[method_name], label=method_name,
                 color=colors[method_name])

    for method_name, recalls in method_recalls.items():
        assert len(normal_threshold) == len(recalls)
        if method_color is not None:
            plt.plot(normal_threshold, recalls, linewidth=3, marker='^', color=method_color[method_name], label=method_name)
        else:
            plt.plot(normal_threshold, recalls, linewidth=3, marker='^', label=method_name, color='#FF6666')

    plt.legend(loc='lower right', fontsize=16)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'normal_recall_%s.png'%(type)))
    else:
        plt.savefig('../results//normal_recall_%s.png'%(type))
    plt.close()
