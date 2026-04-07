import unittest
import os

class TestModel(unittest.TestCase):
    restored_model = None
    base_model = None
    top_model_complete = "model.keras"
    samples_path = "samples"

    def setUp(self):
        from tensorflow.keras.models import load_model
        self.restored_model = load_model(self.top_model_complete)

    def convert_img(self, img):
        import numpy as np
        from tensorflow.keras.utils import load_img, img_to_array
        from tensorflow.keras import applications
        from tensorflow.keras.models import Model
        from tensorflow.image import resize

        # Load image (VGG16 expects 224x224)
        sample_image = load_img(img, target_size=(224, 224))
        sample_img = img_to_array(sample_image) / 255.0
        sample = np.expand_dims(sample_img, axis=0)

        # Load base model
        self.base_model = applications.VGG16(include_top=False, input_shape=(224, 224, 3))

        # Extract features
        layer_name = "block5_pool"
        output_layer = Model(
            inputs=self.base_model.input,
            outputs=self.base_model.get_layer(layer_name).output
        )

        converted_img = output_layer.predict(sample)
        converted_img = converted_img[0]

        # Resize (example: flatten or resize to fixed shape)
        resized_sample = resize(converted_img, (7, 7))
        resized = np.expand_dims(resized_sample, axis=0)

        return resized

    def test_sample1(self):
        sample1 = os.path.join(self.samples_path, "sample1.jpg")

        resized = self.convert_img(sample1)
        result = self.restored_model.predict(resized)

        if result[0][0] > 0.5:
            prediction = "dog"
        else:
            prediction = "cat"

        self.assertEqual(prediction, "dog", "Predicted class is wrong")


if __name__ == "__main__":
    unittest.main()