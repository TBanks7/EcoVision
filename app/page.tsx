"use client";

import React, { useState, useCallback } from 'react';
import { Camera, Upload, Trash2, Trophy, AlertCircle, CheckCircle2, Info, X } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

type WasteAnalysis = {
  type: string;
  recyclable: boolean;
  confidence: number;
  disposalMethod: string;
  environmentalImpact: string;
};

type UserStats = {
  itemsRecycled: number;
  points: number;
  carbonSaved: number;
  streak: number;
};

const mockWasteTypes = [
  {
    type: 'Plastic Bottle',
    recyclable: true,
    disposalMethod: 'Rinse and place in recycling bin',
    environmentalImpact: 'Saves 2.5kg CO2 emissions',
  },
  {
    type: 'Food Container',
    recyclable: true,
    disposalMethod: 'Clean thoroughly before recycling',
    environmentalImpact: 'Reduces landfill waste by 0.5kg',
  },
  {
    type: 'Food Waste',
    recyclable: false,
    disposalMethod: 'Dispose in compost if possible',
    environmentalImpact: 'Can be composted to create nutrient-rich soil',
  },
  {
    type: 'Aluminum Can',
    recyclable: true,
    disposalMethod: 'Crush if possible and recycle',
    environmentalImpact: 'Saves 95% energy vs. new production',
  },
];

const EcoVision = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<WasteAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showTip, setShowTip] = useState(true);
  const [stats, setStats] = useState<UserStats>({
    itemsRecycled: 0,
    points: 0,
    carbonSaved: 0,
    streak: 0,
  });

  const analyzeWaste = useCallback((file: File) => {
    setIsAnalyzing(true);
    
    // Simulate API call
    setTimeout(() => {
      const randomWaste = mockWasteTypes[Math.floor(Math.random() * mockWasteTypes.length)];
      const mockResults: WasteAnalysis = {
        ...randomWaste,
        confidence: Number((Math.random() * 20 + 80).toFixed(1)),
      };

      setAnalysis(mockResults);
      setIsAnalyzing(false);

      if (mockResults.recyclable) {
        setStats(prev => ({
          itemsRecycled: prev.itemsRecycled + 1,
          points: prev.points + 10,
          carbonSaved: prev.carbonSaved + 0.5,
          streak: prev.streak + 1,
        }));
      }
    }, 1500);
  }, []);

  const handleImageUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      analyzeWaste(file);
    }
  }, [analyzeWaste]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 to-white p-4 sm:p-6 md:p-8">
      {showTip && (
        <div className="fixed bottom-4 left-4 right-4 md:hidden bg-white rounded-lg shadow-lg p-4 z-50 animate-fade-in">
          <button 
            onClick={() => setShowTip(false)}
            className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
          >
            <X className="w-4 h-4" />
          </button>
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-1" />
            <p className="text-sm text-gray-600">
              Take a clear photo of your waste item to get instant recycling guidance!
            </p>
          </div>
        </div>
      )}
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 space-y-2">
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-green-600 font-display">
            EcoVision
          </h1>
          <p className="text-gray-600 text-sm md:text-base">
            AI-Powered Waste Recognition & Recycling Assistant
          </p>
        </div>

        {/* Main Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Section */}
          <Card className="overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg md:text-xl">
                <Camera className="w-5 h-5" />
                Waste Scanner
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div 
                  className="w-full aspect-square max-h-[400px] bg-gray-50 rounded-lg relative overflow-hidden border-2 border-dashed border-gray-200 hover:border-green-400 transition-colors"
                >
                  {selectedImage ? (
                    <>
                      <img
                        src={selectedImage}
                        alt="Waste item"
                        className="w-full h-full object-contain"
                      />
                      {isAnalyzing && (
                        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                          <div className="text-white text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-2"></div>
                            <p>Analyzing image...</p>
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center p-4">
                      <Upload className="w-12 h-12 text-gray-400 mb-2" />
                      <p className="text-gray-500 text-center">
                        Upload or take a photo of waste item
                      </p>
                    </div>
                  )}
                </div>
                
                <div className="flex justify-center">
                  <label
                    htmlFor="image-upload"
                    className="inline-flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-full cursor-pointer hover:bg-green-700 transition-colors font-medium"
                  >
                    <Camera className="w-5 h-5" />
                    <span>Take Photo</span>
                  </label>
                  <input
                    type="file"
                    id="image-upload"
                    accept="image/*"
                    capture="environment"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Analysis Results */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg md:text-xl">
                <Trash2 className="w-5 h-5" />
                Analysis Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              {analysis ? (
                <div className="space-y-4">
                  <Alert className={analysis.recyclable ? "bg-green-50 border-green-200" : "bg-yellow-50 border-yellow-200"}>
                    <div className="flex items-start gap-3">
                      {analysis.recyclable ? (
                        <CheckCircle2 className="w-5 h-5 text-green-600 mt-0.5" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
                      )}
                      <div>
                        <AlertTitle className="text-lg font-semibold mb-1">
                          {analysis.type}
                        </AlertTitle>
                        <AlertDescription className="space-y-2">
                          <p className="text-sm">
                            This item is{' '}
                            <span className={analysis.recyclable ? 'text-green-600 font-medium' : 'text-yellow-600 font-medium'}>
                              {analysis.recyclable ? 'recyclable' : 'not recyclable'}
                            </span>
                            <span className="text-gray-500"> • {analysis.confidence}% confidence</span>
                          </p>
                          <div className="bg-white/50 p-3 rounded-md space-y-2 text-sm">
                            <p>
                              <strong>Disposal Method:</strong> {analysis.disposalMethod}
                            </p>
                            <p>
                              <strong>Environmental Impact:</strong> {analysis.environmentalImpact}
                            </p>
                          </div>
                        </AlertDescription>
                      </div>
                    </div>
                  </Alert>
                  {analysis.recyclable && (
                    <div className="animate-fade-in text-center p-3 bg-green-50 rounded-lg">
                      <div className="text-green-600 font-medium">+10 points awarded!</div>
                      <div className="text-sm text-green-500">Keep up the great work!</div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-gray-400 mb-2">
                    <Upload className="w-12 h-12 mx-auto mb-2" />
                    <p>Upload an image to see analysis results</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Impact Stats */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg md:text-xl">
                <Trophy className="w-5 h-5" />
                Your Environmental Impact
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { label: 'Items Recycled', value: stats.itemsRecycled, unit: '' },
                  { label: 'Points Earned', value: stats.points, unit: 'pts' },
                  { label: 'CO₂ Saved', value: stats.carbonSaved.toFixed(1), unit: 'kg' },
                  { label: 'Current Streak', value: stats.streak, unit: 'days' },
                ].map((stat, index) => (
                  <div
                    key={index}
                    className="p-4 bg-gradient-to-br from-green-50 to-green-100/50 rounded-xl text-center"
                  >
                    <div className="text-2xl md:text-3xl font-bold text-green-600 mb-1">
                      {stat.value}{stat.unit}
                    </div>
                    <div className="text-sm text-gray-600">{stat.label}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default EcoVision;