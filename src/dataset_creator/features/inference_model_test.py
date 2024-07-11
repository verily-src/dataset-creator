"""Tests for inference_model.py."""

import tensorflow as tf

from dataset_creator.features import inference_model


class InferenceModelTest(tf.test.TestCase):

  def test_model_inference_from_functional_layer(self):
    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV3Small(weights=None),
    ])
    new_model = inference_model.inference_model(model, ['MobilenetV3small'])
    input_tensor = tf.random.normal((1, 224, 224, 3), dtype=tf.float32)
    model_output = model(input_tensor)
    new_model_output = new_model(input_tensor)['MobilenetV3small']
    self.assertAllClose(model_output, new_model_output)

  def test_model_inference_with_multiple_outputs(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((224, 224, 3), name='input'),
        tf.keras.layers.Lambda(lambda x: x + tf.ones_like(x), name='add_1'),
        tf.keras.layers.Lambda(lambda x: x + tf.ones_like(x), name='add_2'),
    ])
    new_model = inference_model.inference_model(model, ['add_1', 'add_2'])
    input_tensor = tf.random.normal((1, 224, 224, 3), dtype=tf.float32)
    model_output = new_model(input_tensor)
    self.assertIn('add_1', model_output)
    self.assertIn('add_2', model_output)

  def test_model_inference_with_multiple_inputs_of_different_depths(self):
    input_1 = tf.keras.layers.Input((10,))
    depth_1_input = tf.keras.layers.Identity()(input_1)
    input_2 = tf.keras.layers.Input((5,))

    # Concatenate layer has a 0-depth input, and a depth 1 input.
    concatenate_layer = tf.keras.layers.Concatenate(axis=-1, name='concatenate')
    concatenated = concatenate_layer([depth_1_input, input_2])

    model = tf.keras.Model(inputs=[input_1, input_2], outputs=[concatenated])
    new_model = inference_model.inference_model(model, ['concatenate'])
    model_output = new_model([tf.zeros((1, 10)), tf.zeros((1, 5))])
    self.assertIn('concatenate', model_output)
    self.assertEqual(model_output['concatenate'].shape, (1, 15))

  def test_model_inference_with_missing_output(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((224, 224, 3), name='input'),
        tf.keras.layers.Lambda(lambda x: x + tf.ones_like(x), name='add_1'),
        tf.keras.layers.Lambda(lambda x: x + tf.ones_like(x), name='add_2'),
    ])
    with self.assertRaisesRegex(
        ValueError, r"^add_3, add_4 are not present in the model's layers"):
      inference_model.inference_model(model, ['add_1', 'add_3', 'add_4'])
