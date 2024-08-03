import os
import cv2

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tiff', '.TIFF', '.tif']

def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def get_all_images(path, n_folders=0, sort_function=None):
	"""
	return all images in the folder
	:param path: path to Data Folder, absolute path
	:return: 1D list of image files absolute path
	"""
	# TODO: Tail Call Elimination
	abs_path = os.path.abspath(path) # Returns absolute path of passed directory or file
	image_files = list()
	# num = 0
	if os.path.isfile(abs_path):
		return [abs_path]
	else:
		for subpath in os.listdir(abs_path): # to get the list of all files and directories in the specified directory
			if _is_image_file(subpath):
				# num = num +1
				image_files.append(os.path.join(abs_path, subpath))
			elif n_folders > 0:
				if os.path.isdir(os.path.join(abs_path, subpath)):
					image_files = image_files + get_all_images(os.path.join(abs_path, subpath), n_folders - 1)
		image_files.sort(key=sort_function)
		return image_files

def save_image_in_path(file_path, image):
    # Extract the directory part of the file path
    directory = os.path.dirname(file_path)
    
    # Create directories if they don't exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    cv2.imwrite(file_path, image)
    