import cv2

# Load the mask and scene images
mask = cv2.imread('/home/stefan/PycharmProjects/ZS6D/test/drill/000001_000004.png', cv2.IMREAD_GRAYSCALE)
scene = cv2.imread('/home/stefan/PycharmProjects/ZS6D/test/drill/000001.png')

# Ensure mask is binary (black and white only)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Create a 3-channel mask
mask_3channel = cv2.merge([mask, mask, mask])

# Apply the mask to the scene
result = cv2.bitwise_and(scene, mask_3channel)

# Save the result
cv2.imwrite('/home/stefan/PycharmProjects/ZS6D/test/drill/0.png', result)

print("Image processing complete. Result saved as 'result.png'.")