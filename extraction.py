import os
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pydicom
from PIL import Image
import matplotlib.pyplot as plt


class DICOMAnnotationDataset(Dataset):
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root (str): Root directory containing the 'CP#' folders.
                To get this to work correct, open your commmand prompt by pressing the windows key, searching "cmd", and opening it
                From here, you want to write "dir" and press enter, this will open your directory
                From here, you want to follow the path to the EnFace Images folder by typed "cd" followed by the next folder in the path (P.S. You can press tab to autocomplete)
                Get to the EnFace Images folder and cd into it using the tab autocomplete
                Once you are in it, copy the path directly from the command prompt and paste it in between the "" for DATA_ROOT (make sure you dont delete the r)
                We have to do this bc there is something funky happening with the folder name where I think it has weird hidden characters, this way we are able to caputer them so the system can find the file
            transform (callable, optional): Optional transform to apply to both images and DICOMs.
        """
        self.data_root = data_root
        self.transform = transform
        self.dataset = self._collect_data()

    def _collect_data(self):
        """Collects the (DICOM, PNG) file paths."""
        dataset = []
        
        # Traverse all CP# folders
        for folder in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder)

            if os.path.isdir(folder_path) and folder.startswith("CP"):
                dicom_folder = os.path.join(folder_path, "DICOMs")
                annotation_mask = os.path.join(folder_path, "AnnotationMask_Malign.png")

                # Ensure the DICOM folder and annotation mask exist
                if os.path.exists(dicom_folder) and os.path.exists(annotation_mask):
                    dicom_file = os.path.join(dicom_folder, "EF_Total.dcm")

                    # Add to the dataset if both the DICOM and annotation mask exist
                    if os.path.exists(dicom_file):
                        dataset.append((dicom_file, annotation_mask))

        return dataset

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Loads and returns a sample from the dataset."""
        dicom_path, png_path = self.dataset[idx]

        # Load the DICOM file and extract the pixel data
        dicom_data = pydicom.dcmread(dicom_path).pixel_array
        dicom_tensor = torch.tensor(dicom_data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Load the PNG annotation mask
        mask = Image.open(png_path).convert("L")  # Convert to grayscale
        mask_tensor = ToTensor()(mask)

        # Apply any optional transforms
        if self.transform:
            dicom_tensor = self.transform(dicom_tensor)
            mask_tensor = self.transform(mask_tensor)

        return dicom_tensor, mask_tensor

# Helper function to display DICOM and PNG images side by side
def show_pair(dicom, mask, idx):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the DICOM image
    axs[0].imshow(dicom.squeeze(0), cmap="gray")  # Remove the channel dimension for display
    axs[0].set_title(f"DICOM Image {idx+1}")
    axs[0].axis("off")

    # Display the Annotation Mask
    axs[1].imshow(mask.squeeze(0), cmap="gray")  # Remove the channel dimension for display
    axs[1].set_title(f"Annotation Mask {idx+1}")
    axs[1].axis("off")

    plt.show()

# Instantiate the dataset
DATA_ROOT = r"C:\Users\phill\Files\Miscellaneous\Gabrielle_Python\EnFace Images (Spectral-15Band) - DICOM"
dataset = DICOMAnnotationDataset(DATA_ROOT)

# Use DataLoader for batching and shuffling
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Example: Iterate through the DataLoader
for dicom, mask in dataloader:
    print(f"DICOM shape: {dicom.shape} | Mask shape: {mask.shape}")

for i, (dicom, mask) in enumerate(dataloader):
    if i >= 10:
        break
    print(f"Displaying pair {i+1}\n")
    show_pair(dicom[0], mask[0], i)
