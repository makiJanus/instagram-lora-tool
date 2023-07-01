import os
import instaloader
import os
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# Functions 
class tools():
    def __init__(self):
        """
        Initialize the image captioning model.
        """
        # Processor and model variables for the class
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
        
    def scrape_instagram_images(self, username, max_images=None):
        """
        Scrape Instagram images for a given username.

        Args:
            username (str): The username of the Instagram profile.
            max_images (int): The maximum number of images to scrape (optional).

        Returns:
            None
        """
        # Create a directory to store the scraped images
        dir = f"scrapped/{username}"
        os.makedirs(dir, exist_ok=True)
        
        # Create an instance of the Instaloader class
        loader = instaloader.Instaloader(dirname_pattern=dir)

        # Retrieve the profile metadata for the specified username
        try:
            profile = instaloader.Profile.from_username(loader.context, username)
        except instaloader.exceptions.ProfileNotExistsException:
            print(f"Profile '{username}' does not exist or is not accessible.")
            return    

        # Iterate over the profile's posts and download the images
        count = 0  # Counter for the number of downloaded images
        for post in profile.get_posts():
            # Skip posts that are not images
            if not post.is_video:
                # Download the image
                try:
                    loader.download_post(post, target=os.path.join(dir))
                    print(f"Downloaded image: {post.date_utc}")
                    count += 1

                    # Break the loop if the maximum number of images is reached
                    if max_images is not None and count >= max_images:
                        break
                except Exception as e:
                    print(f"Failed to download image: {post.date_utc}. Error: {str(e)}")

        print("Scraping completed successfully.")


    def get_folder_list(self, path):
        """
        Get a list of subdirectories (folders) in the specified path.

        Args:
            path (str): The path to the directory.

        Returns:
            list: A list of subdirectory names.

        """
        # Get a list of subdirectories (folders) in the specified path
        folder_list = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return folder_list

    def select_folder(self, folder):
        """
        Process the selected folder and perform any desired actions with it.

        Args:
            folder (str): The name of the selected folder.

        Returns:
            None
        """
        # Process the selected folder
        print(f"Selected folder: {folder}")

    def process_images(self, username):
        """
        Process the images in the specified user's folder by detecting faces, cropping around the largest face,
        and resizing the cropped image.

        Args:
            username (str): The username associated with the images.

        Returns:
            None

        """
        # Size trainning size for kohya Kohya
        new_size = (512,512)
        # Folder where the scrapped images are
        input_folder = f"scrapped/{username}"
        # Create the output folder if it doesn't exist
        output_folder = f"scrapped/{username}/cropped_centered"
        os.makedirs(output_folder, exist_ok=True)

        # Load the pre-trained face detector from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Iterate over the images in the input folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.jpg', '.png')):
                # Read the image
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                # Convert the image to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                # If there are faces in image
                if len(faces) > 0:
                    # Get the largest detected face
                    (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])

                    # Calculate the center of the face
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Calculate the coordinates for cropping the image around the face
                    half_size = min(image.shape[0], image.shape[0]) // 2
                    x1 = max(center_x - half_size, 0)
                    y1 = max(center_y - half_size, 0)
                    x2 = min(center_x + half_size, image.shape[1])
                    y2 = min(center_y + half_size, image.shape[0])

                    # Crop the image around the face
                    cropped_image = image[y1:y2, x1:x2]

                    # Resize the cropped image
                    resized_image = cv2.resize(cropped_image, new_size)

                    # Save the processed image to the output folder
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, resized_image)

    def blip_captioning(self, username):
        """
        Generate descriptive captions for the images in the specified user's folder using the BLIP Hugging Face model.
        The captions are stored in separate text files with the same name as the corresponding image.

        Args:
            username (str): The username associated with the images.

        Returns:
            None

        """
        # Select input folder
        input_folder = f"scrapped/{username}/cropped_centered"
        
        # Create the output folder if it doesn't exist
        output_folder = f"scrapped/{username}/cropped_centered"
        os.makedirs(output_folder, exist_ok=True)
        
        # For every image (png or jpg) in input_folder:
        for image_file in os.listdir(input_folder):
            if image_file.endswith(".png") or image_file.endswith(".jpg"):
                # Add images files to list of paths varaible
                input_path = f"scrapped/{username}/cropped_centered"
                print("input_path: " + input_path)
                
                image_path = os.path.join(input_path, image_file)
                # Replace backslashes with forward slashes
                image_path = image_path.replace("\\", "/")
                
                print("image_path: " + image_path)
                
                # Generate descriptive captioning with BLIP Hugging Face
                blip_caption = self.generate_blip_caption(username ,image_path)
                
                # Generate a txt file with the same name as the image
                txt_filename = os.path.splitext(image_file)[0] + ".txt"
                txt_filepath = os.path.join(output_folder, txt_filename)
                
                # Store the BLIP captioning in the corresponding txt file,
                # always using the username variable as the first word
                if os.path.exists(txt_filepath):
                    with open(txt_filepath, "w") as txt_file:
                        # Append caption to the existing file
                        txt_file.write(f"{blip_caption}\n")
                else:
                    with open(txt_filepath, "w") as txt_file:
                        # Create a new file with the caption
                        txt_file.write(f"{blip_caption}\n")  

    def generate_blip_caption(self, username, image_path):
        """
        Generate a descriptive caption for the specified image using the BLIP Hugging Face model.

        Args:
            username (str): The username associated with the image.
            image_path (str): The path to the image.

        Returns:
            str: The generated caption for the image.

        """
        # Load the image
        raw_image = Image.open(image_path).convert('RGB')
        
        # Conditional image captioning
        text = f"a image of {username}, "
        inputs = self.processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)
        
        max_length = 100
        out = self.model.generate(**inputs, max_length=max_length)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption
        
    def change_image_blip_next(self, username, index_image):
        """
        Change the selected image and corresponding text for the BLIP Hugging Face model to the next image in the sequence.

        Args:
            username (str): The username associated with the images.
            index_image (int): The current index of the selected image.

        Returns:
            tuple: A tuple containing the path of the selected image, the updated index, and the text associated with the image.

        """
        # Select input folder
        input_folder = f"scrapped/{username}/cropped_centered"
        
        # Get the list of image paths in the input folder
        image_paths = [
            os.path.join(input_folder, image_file) 
            for image_file in os.listdir(input_folder) 
            if image_file.endswith(('.png', '.jpg'))
        ]
        txt_paths = [
            os.path.join(input_folder, txt_file) 
            for txt_file in os.listdir(input_folder) 
            if txt_file.endswith(('.txt'))
        ]
        
        index_image += 1
        
        if 0 <= index_image < len(image_paths):
            # Get the image path of the specified index
            selected_image_path = image_paths[index_image]
            # Read the text from the selected TXT file
            selected_txt_path = txt_paths[index_image]
            with open(selected_txt_path, "r") as txt_file:
                text = txt_file.read()
        
        else:
            # If the index is out of range, reset the index to 0
            index_image = 0
            selected_image_path = image_paths[index_image]
            selected_txt_path = txt_paths[index_image]
            # Read the text from the selected TXT file
            with open(selected_txt_path, "r") as txt_file:
                text = txt_file.read()
        
            
        # Replace backslashes with forward slashes in the file path
        selected_image_path = selected_image_path.replace("\\", "/")
        
        return selected_image_path, index_image, text

    def change_image_blip_previous(self, username, index_image):
        """
        Change the selected image and corresponding text for the BLIP Hugging Face model to the next image in the sequence.

        Args:
            username (str): The username associated with the images.
            index_image (int): The current index of the selected image.

        Returns:
            tuple: A tuple containing the path of the selected image, the updated index, and the text associated with the image.

        """
        # Select input folder
        input_folder = f"scrapped/{username}/cropped_centered"
        
        # Get the list of image paths in the input folder
        image_paths = [
            os.path.join(input_folder, image_file) 
            for image_file in os.listdir(input_folder) 
            if image_file.endswith(('.png', '.jpg'))
        ]
        txt_paths = [
            os.path.join(input_folder, txt_file) 
            for txt_file in os.listdir(input_folder) 
            if txt_file.endswith(('.txt'))
        ]
        
        index_image -= 1
        
        if 0 <= index_image < len(image_paths):
            # Get the image path of the specified index
            selected_image_path = image_paths[index_image]
             # Read the text from the selected TXT file
            selected_txt_path = txt_paths[index_image]
            with open(selected_txt_path, "r") as txt_file:
                text = txt_file.read()
        
        else:
            # If the index is out of range, reset the index to 0
            index_image = len(image_paths)-1
            selected_image_path = image_paths[index_image]
            selected_txt_path = txt_paths[index_image]
            # Read the text from the selected TXT file
            with open(selected_txt_path, "r") as txt_file:
                text = txt_file.read()
        
            
        # Replace backslashes with forward slashes in the file path
        selected_image_path = selected_image_path.replace("\\", "/")
        
        return selected_image_path, index_image, text

    def change_caption(self, username, caption, index_image):
        """
        Change the caption of the selected image.

        Args:
            username (str): The username associated with the images.
            caption (str): The new caption to be assigned to the selected image.
            index_image (int): The index of the selected image.

        """
        # Select input folder
        input_folder = f"scrapped/{username}/cropped_centered"
        
        # Get the list of txt paths in the input folder
        txt_paths = [
            os.path.join(input_folder, txt_file) 
            for txt_file in os.listdir(input_folder) 
            if txt_file.endswith(('.txt'))
        ]
        
        # Read the text from the selected TXT file
        selected_txt_path = txt_paths[index_image]
        
        with open(selected_txt_path, "w") as txt_file:
            # Create a new file with the caption
            txt_file.write(f"{caption}\n")