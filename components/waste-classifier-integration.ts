"use client";

import * as tf from '@tensorflow/tfjs';
import { useState, useEffect, useCallback } from 'react';

// Types remain the same
interface WasteClassification {
  category: string;
  confidence: number;
  details: {
    recyclable: boolean;
    disposalMethod: string;
    environmentalImpact: string;
  };
}

const CATEGORY_MAPPING = {
  0: 'paper',
  1: 'containers',
  2: 'compost',
  3: 'waste'
} as const;

const CATEGORY_DETAILS = {
  paper: {
    recyclable: true,
    disposalMethod: 'Paper recycling bin',
    environmentalImpact: 'Saves trees and reduces landfill waste',
    items: ['coffee cups (no lids)', 'office paper', 'newspaper', 'magazines', 'cardboard']
  },
  containers: {
    recyclable: true,
    disposalMethod: 'Container recycling bin',
    environmentalImpact: 'Reduces plastic waste and saves energy',
    items: ['plastic bottles', 'glass bottles', 'aluminum cans', 'milk cartons']
  },
  compost: {
    recyclable: true,
    disposalMethod: 'Compost bin',
    environmentalImpact: 'Creates nutrient-rich soil and reduces methane emissions',
    items: ['food waste', 'paper towels', 'compostable containers']
  },
  waste: {
    recyclable: false,
    disposalMethod: 'Waste bin',
    environmentalImpact: 'Cannot be recycled - goes to landfill',
    items: ['coffee cup lids', 'straws', 'chip bags']
  }
} as const;

export const useWasteClassifier = () => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    initializeModel();
  }, []);

  const initializeModel = async () => {
    try {
      await tf.ready();
      const model = createAndTrainModel();
      setModel(model);
      setIsLoading(false);
    } catch (err) {
      console.error('Error initializing model:', err);
      setError('Failed to initialize model');
      setIsLoading(false);
    }
  };

  const createAndTrainModel = () => {
    // Create a model that expects image input
    const model = tf.sequential();
    
    // Add layers that can handle image input (224x224x3)
    model.add(tf.layers.conv2d({
      inputShape: [224, 224, 3],
      kernelSize: 3,
      filters: 32,
      activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.conv2d({
      kernelSize: 3,
      filters: 64,
      activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    // Flatten the 2D image to 1D array
    model.add(tf.layers.flatten());
    
    // Dense layers for classification
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 4, activation: 'softmax' })); // 4 classes

    // Compile the model
    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    return model;
  };

  const preprocessImage = async (imageData: ImageData): Promise<tf.Tensor4D> => {
    return tf.tidy(() => {
      console.log('Original image tensor shape:', imageData.data.length);
      
      // Convert to tensor
      const tensor = tf.browser.fromPixels(imageData);
      console.log('Image tensor shape:', tensor.shape);
      
      // Resize to match model input size
      const resized = tf.image.resizeBilinear(tensor, [224, 224]);
      console.log('Resized tensor shape:', resized.shape);
      
      // Normalize values to [0, 1]
      const normalized = resized.toFloat().div(tf.scalar(255));
      
      // Add batch dimension
      const batched = normalized.expandDims(0) as tf.Tensor4D;
      console.log('Final preprocessed tensor shape:', batched.shape);
      
      return batched;
    });
  };

  const classifyImage = async (imageData: ImageData): Promise<WasteClassification> => {
    if (!model) {
      throw new Error('Model not loaded');
    }

    try {
      // Preprocess image
      const tensor = await preprocessImage(imageData);
      console.log('Input tensor shape:', tensor.shape);

      // Run prediction
      const predictions = model.predict(tensor) as tf.Tensor;
      const probabilities = await predictions.data();
      
      // Get predicted class
      const maxProbIndex = probabilities.indexOf(Math.max(...Array.from(probabilities)));
      const predictedCategory = CATEGORY_MAPPING[maxProbIndex as keyof typeof CATEGORY_MAPPING];
      const confidence = probabilities[maxProbIndex] * 100;

      // Cleanup
      tensor.dispose();
      predictions.dispose();

      return {
        category: predictedCategory,
        confidence,
        details: CATEGORY_DETAILS[predictedCategory]
      };
    } catch (err) {
      console.error('Classification error:', err);
      throw new Error(`Failed to classify image: ${err}`);
    }
  };

  return {
    classifyImage,
    isLoading,
    error,
    modelLoaded: !!model
  };
};


// Helper function remains the same
export const getImageDataFromFile = async (file: File): Promise<ImageData> => {
  return new Promise((resolve, reject) => {
    if (!file.type.startsWith('image/')) {
      reject(new Error('Invalid file type. Please provide an image file.'));
      return;
    }

    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        reject(new Error('Failed to get canvas context'));
        return;
      }

      ctx.drawImage(img, 0, 0);
      try {
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        console.log('Image data extracted:', {
          width: imageData.width,
          height: imageData.height
        });
        resolve(imageData);
      } catch (err) {
        reject(new Error('Failed to extract image data: ' + (err instanceof Error ? err.message : 'Unknown error')));
      }
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to load image'));
    };

    img.src = url;
  });
};