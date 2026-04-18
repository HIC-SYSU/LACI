from glob import glob
import SimpleITK as sitk
from tqdm import tqdm


def nrrd_to_nifti(nrrd_path, nifti_path):
    # 读取NRRD文件
    image = sitk.ReadImage(nrrd_path)
    # 将读取的影像保存为NIfTI格式
    sitk.WriteImage(image, nifti_path, True)  # True表示保存为压缩的.nii.gz格式

def nifti_to_nrrd(nifti_path, nrrd_path):
    # 读取NIfTI文件
    image = sitk.ReadImage(nifti_path)
    # 将读取的影像保存为NRRD格式
    sitk.WriteImage(image, nrrd_path)


if __name__ == "__main__":
    # listt = glob(
    #     '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/*/lgemri.nrrd')
    # for item in tqdm(listt):
    #     nifti_path = item.replace('.nrrd', '.nii.gz')
    #     label_nrrd_path = item.replace('lgemri', 'laendo')
    #     label_nii_path = label_nrrd_path.replace('.nrrd', '.nii.gz')
    #     # nrrd_to_nifti(item, nifti_path)
    #     nrrd_to_nifti(label_nrrd_path, label_nii_path)

    # listt = glob(
    #     '/data/chenjinfeng/40180-data/chenjinfeng/Data_download/cardiac/2018_UTAH_MICCAI/Training Set/*/lgemri.nii.gz')
    # for item in tqdm(listt):
    #     nifti_path = item.replace('.nii.gz', '_i2d.nrrd')
    #     label_nrrd_path = item.replace('lgemri', 'laendo')
    #     label_nii_path = label_nrrd_path.replace('.nii.gz', '_i2d.nrrd')
    #     # nrrd_to_nifti(item, nifti_path)
    #     nrrd_to_nifti(label_nrrd_path, label_nii_path)

    # nii_path = glob('/data/chenjinfeng/Data/CT_160/Xinan/image_CT/*.nii.gz')
    # for item in tqdm(nii_path):
    #     nifti_path = item.replace('.nii.gz', '.nrrd')
    #     nirrd_path = nifti_path.replace('image_CT', 'image_CT_nrrd')
    #     nrrd_to_nifti(item, nirrd_path)

    # nii_path = glob('/data/chenjinfeng/Data/CT_160/Xinan/label_CT/*.nii.gz')
    # for item in tqdm(nii_path):
    #     nifti_path = item.replace('.nii.gz', '.nrrd')
    #     nirrd_path = nifti_path.replace('label_CT', 'label_CT_nrrd')
    #     nrrd_to_nifti(item, nirrd_path)

    nii_path = glob('/data/chenjinfeng/Data/CT_160/Xinan/image_CT_nrrd/*.nrrd')
    for item in tqdm(nii_path):
        nifti_path = item.replace('.nrrd', '.nii.gz')
        nirrd_path = nifti_path.replace('image_CT_nrrd', 'nii')
        nrrd_to_nifti(item, nirrd_path)


