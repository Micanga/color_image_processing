import imageio
import numpy as np
from skimage.color import rgb2hsv
from skimage.morphology import disk, opening, closing, dilation, erosion

######
# METHODS CODE
######
def channels_opening(rgb_img,k):
	img = np.zeros((rgb_img.shape[0],rgb_img.shape[1],3))
	img[:,:,0] = opening(rgb_img[:,:,0], disk(k))
	img[:,:,1] = opening(rgb_img[:,:,1], disk(k))
	img[:,:,2] = opening(rgb_img[:,:,2], disk(k))
	return img

def comp_operation(rgb_img,k):
	# 1. Converting the RGB to HSV
	hsv_img = rgb2hsv(rgb_img)

	# 2. Normalising the H channel to the interval 0 - 255
	h = hsv_img[:,:,0]
	min_i, max_i = np.min(h), np.max(h)
	for i in range(len(h)):
		for j in range(len(h[i])):
			pixel = h[i][j]
			h[i][j] = \
				int((pixel-min_i)*(255/(max_i-min_i)))

	# 3. Performing the morphological gradient with the structuring element disk
	morph_grad = dilation(h, disk(k)) \
					- erosion(h, disk(k))

	# 4. Normalising the resulting morphological gradient to the interval 0 - 255
	min_i, max_i = np.min(morph_grad), np.max(morph_grad)
	for i in range(len(morph_grad)):
		for j in range(len(morph_grad[i])):
			pixel = morph_grad[i][j]
			morph_grad[i][j] = \
				int((pixel-min_i)*(255/(max_i-min_i)))

	# 5. Composing the new RGB image
	img = np.zeros((rgb_img.shape[0],rgb_img.shape[1],3))
	opening_img = opening(h, disk(k))
	closing_img = closing(h, disk(k))
	for i in range(len(img)):
		for j in range(len(img[i])):
			img[i][j][0] = morph_grad[i][j]
			img[i][j][1] = opening_img[i][j]
			img[i][j][2] = closing_img[i][j]
	return img

#####
# METRIC METHOD
#####
def RMSE(input_img,output_img):
	width, height, depth = len(input_img),len(input_img[0]),len(input_img[0][0])
	RMSE = np.sum((np.array(input_img).astype(float) - np.array(output_img).astype(float))**2)
	RMSE /= (width*height)
	return np.sqrt(RMSE)

######
# MAIN CODE
######
# 1. Reading the inputs and defining the variables
# a. image filename, k (size of the structuring element)
# option = 1 for RGB opening, 2 = for composition of operations
filename = str(input()).rstrip()
k = int(input())
option  = int(input())

# 2. Starting the option process
if option == 1:
	original_img = imageio.imread(filename)
	result_img = channels_opening(original_img,k)
elif option == 2:
	original_img = imageio.imread(filename)
	result_img = comp_operation(original_img,k)
else:
	original_img = imageio.imread(filename)
	result_img = channels_opening(original_img,2*k)
	result_img = comp_operation(result_img,k)

# 3. Printing the RSE result between the original image and the result
print("%.4f" % RMSE(original_img,result_img))