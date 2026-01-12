import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, GlobalAveragePooling2D, Dense, multiply,
    Activation, Reshape, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def attention_module(net, attention_name='attention'):
    """
    Self-attention module for a CNN.
    """
    # Squeeze: Global Average Pooling to get channel-wise statistics
    squeeze = GlobalAveragePooling2D()(net)

    # Excitation: Two fully connected layers to learn channel-wise weights
    excitation = Dense(units=net.shape[-1] // 16, activation='relu')(squeeze)
    excitation = Dense(units=net.shape[-1], activation='sigmoid')(excitation)

    # Reshape weights to match the input feature map
    excitation = Reshape((1, 1, net.shape[-1]))(excitation)

    # Scale: Multiply the input feature map by the learned weights
    scale = multiply([net, excitation], name=attention_name)

    return scale

def create_ha_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Create the Hybrid Attention-CNN (HA-CNN) model.
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # Pre-trained backbone (ResNet50)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Get the output of the backbone
    x = base_model.output

    # Apply the attention module
    x = attention_module(x)

    # Classifier head
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

if __name__ == '__main__':
    # Example of creating the model
    model = create_ha_cnn_model()
    model.summary()
