import time
import pygame
import cv2
import numpy as np

from dataclasses import dataclass
from PIL import Image
from pygame.locals import KEYDOWN, K_ESCAPE, K_p, K_r, QUIT
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from emotion_detection.utils import parse_args, get_val_transform
from emotion_detection.dataset_handler import FERDataset


@dataclass
class WebcamVideo:
	'''Webcam class with a UI and pause/record states, using Pygame + openCV.
	'''
	# Dimensions of the entire containing window
	window_dims: tuple = (720, 480)
	# Dimensions of the video, within the containing window
	video_dims: tuple = (425, 425)
	# Pygame font attributes
	font_type: str = 'monospace'
	font_size: int = 15
	font_color: tuple = (250, 250, 250)

	def __post_init__(self):
		# Setup Pygame
		pygame.init()
		pygame.display.set_caption('OpenCV Pygame camera video stream')
		# Set the camera (i.e., webcam) to use for video streaming
		self.camera = cv2.VideoCapture(0)
		# Set the Pygame screen (i.e., display) and font
		self.screen = pygame.display.set_mode(self.window_dims)
		self.font = pygame.font.SysFont(self.font_type, self.font_size)
		# Possible video states
		self.running = True
		self.paused = False
		self.recording = False

	def refresh(self):
		'''Refreshes/updates the entire Pygame display.
		'''
		pygame.display.flip()

	def shutdown(self):
		'''Shuts down the video (i.e., Pygame and openCV displays/windows)
		'''
		pygame.quit()
		cv2.destroyAllWindows()

	def handle_key_events(self):
		'''Updates running/paused/recording statuses from list of key events.
		'''
		events = pygame.event.get()
		for e in events:
			if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
				self.running = False
			elif (e.type == KEYDOWN and e.key == K_p):
				self.paused = not self.paused
			elif (e.type == KEYDOWN and e.key == K_r):
				self.recording = not self.recording

	def images_from_camera(self):
		'''Reads the current webcam frame and returns it as RGB and gray images.

		Returns:
			cv2.Mat, cv2.Mat: RGB and gray image matrices, respectively
		'''
		# Read the camera frame and convert to a colored surface array
		_, frame = self.camera.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = np.rot90(frame)
		frame = pygame.surfarray.make_surface(frame)
		# Create the RGB and gray image matrices
		rgb_image = pygame.surfarray.array3d(frame)
		rgb_image = np.rot90(rgb_image, -1)
		gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
		return rgb_image, gray_image

	def update_video(self, video_frame, coords=(0, 0)):
		'''Updates the video with an image at certain (x, y) coordinates.

		Args:
			video_frame (cv2.Mat): The new/updated video frame
			coords (tuple, optional): (x, y) coordinates. Defaults to (0, 0).
		'''
		snapshot = pygame.surfarray.make_surface(np.rot90(video_frame))
		snapshot = pygame.transform.scale(snapshot, self.video_dims)
		self.screen.blit(snapshot, coords)

	def update_text(self, text, coords=(0, 0)):
		'''Updates the window with text at certain (x, y) coordinates.

		Args:
			text (str): The text to add to the window
			coords (tuple, optional): (x, y) coordinates. Defaults to (0, 0).
		'''
		font_text = self.font.render(text, 1, self.font_color)
		self.screen.blit(font_text, coords)


class LiveEmotionDetector:
	'''Class to detect facial expressions from a live webcam
	'''
	def __init__(
			self,
			emotion_detection_model,
			feature_extractor,
			target_transform,
			face_detection_model,
			device,
	):
		# Model to perform emotion detection/classification on a facial image
		self.emotion_detection_model = emotion_detection_model
		# Feature extractor to get pretrained model metadata (e.g., input shape)
		self.feature_extractor = feature_extractor
		# Target transform applied to images fed to the emotion detection model
		self.target_transform = target_transform
		# Face cascade model to detect/extract faces from a camera frame
		self.face_detection_model = face_detection_model
		# The PyTorch device to use for inference
		self.device = device

	def run(self, window_dims, video_dims, font_type='monospace', font_size=15):
		'''Runs the emotion detection demo.

		Args:
			window_dims (tuple): Dimensions (w, h), in pixels, of demo window
			video_dims (_type_): Dimensions (w, h), in pixels, of webcam video
			font_type (str, optional): Pygame font type. Defaults to 'monospace'.
			font_size (int, optional): Pygame font size. Defaults to 15.
		'''
		video = WebcamVideo(
			window_dims=window_dims,
			video_dims=video_dims,
			font_type=font_type,
			font_size=font_size,
		)

		while video.running:
			# Check and handle user key presses
			video.handle_key_events()
			if video.paused:
				continue
			# Start timer (to measure FPS) and fill the screen w/ black pixels
			start_time = time.time()
			video.screen.fill([0, 0, 0])
			# Get RGB and gray images from the current webcam frame
			rgb_image, gray_image = video.images_from_camera()
			# Detect all faces in the (grayscale) webcam frame
			faces = self.face_detection_model.detectMultiScale(gray_image, 1.3, 5)
			# Check if any faces were detected
			if isinstance(faces, np.ndarray):
				# Extract and show the (first) detected face
				extracted_face = self.face_from_image(faces[0], rgb_image)
				video.update_video(extracted_face)
			else:
				extracted_face = None
			# Predict and show the emotion of the extracted face
			emotion = self.emotion_from_face(extracted_face)
			video.update_text(emotion, (video_dims[0] / 2, video_dims[1]))
			# Calculate the demo video's frames per second
			fps = 1 / (time.time() - start_time)
			video.update_text('{0:.1f} FPS'.format(fps))
			# Refresh the video to persist the new data
			video.refresh()
		video.shutdown()

	def face_from_image(self, face_coords, image_with_face):
		'''Extracts a face from an image, given the face's coordinates.

		Args:
			face_coords (tuple): The face's (x, y, w, h) coordinates
			image_with_face (cv2.Mat): The image containing the face to extract

		Returns:
			cv2.Mat: The extracted (resized) face
		'''
		(x, y, w, h) = face_coords
		extracted_face = image_with_face[y:y + h, x:x + w]
		image_dims = (self.feature_extractor.size, self.feature_extractor.size)
		return cv2.resize(extracted_face, image_dims)

	def emotion_from_face(self, input_face):
		'''Given an extracted face as input, predicts its emotion

		Args:
			input_face (cv2.Mat): The face to predict an emotion from

		Returns:
			str: The predicted emotion
		'''
		# If there is no face, assume it was not detected in the video
		if input_face is None:
			return 'Face not detected'
		# Prepare the image and pass it to the model
		facial_image = Image.fromarray(input_face).convert('RGB')
		model_input = self.target_transform(facial_image).unsqueeze(0)
		model_output = self.emotion_detection_model(model_input)
		# Get the most likely emotion based on the model's output
		logits = model_output.logits.to(self.device)
		predictions = logits.argmax(-1).to(self.device)
		return FERDataset.CLASSES[int(predictions[0])]


def main(args):
	'''Runs the live emotion detection demo using the CLI arguments

	Args:
		args (Namespace): The script's parsed arguments
	'''
	# Model for emotion detection/FER
	model = ResNetForImageClassification.from_pretrained(args.output_dir)
	# Feature extractor corresponding to the pretrained model
	feature_extractor = AutoFeatureExtractor.from_pretrained(args.pretrained_model_name)
	# Target image transform for the emotion detection/FER model
	target_transform = get_val_transform(feature_extractor)
	# Cascade classifier to detect (get the coordinates of) human faces
	face_cascade = cv2.CascadeClassifier(args.demo_cascade_file)
	# Create and run the live emotion detector
	emotion_detector = LiveEmotionDetector(
		emotion_detection_model=model,
		feature_extractor=feature_extractor,
		target_transform=target_transform,
		face_detection_model=face_cascade,
		device=args.device,
	)
	emotion_detector.run(
		window_dims=(args.demo_window_width, args.demo_window_height),
		video_dims=(args.demo_video_width, args.demo_video_height),
		font_type=args.demo_font_type,
		font_size=args.demo_font_size,
	)


if __name__ == '__main__':
	# Parse the script arguments and run the demo
	args = parse_args()
	main(args)
