import matplotlib.pyplot as plt
from PIL import Image
import io
import matplotlib.image as mpimg
from utils.transforms import training_img_transforms
import matplotlib.pyplot as plt
import io


img = Image.open('/home/zhaoxizh/zhaoxizh_project/uflow/files/00000.jpg')
buf = io.BytesIO()

# Save the image into the buffer.
img = training_img_transforms(img)

image_np = img.permute(1, 2, 0).detach().numpy()

plt.imshow(image_np)
plt.axis('off')  # Remove axes

# Save the plot to a BytesIO buffer as PNG
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)  # Move the buffer position to the start

# Display the image from the BytesIO buffer
img_png = buf.getvalue()
with open('example_plot.txt', 'wb') as f:
    f.write(img_png)


