import AsyncStorage from '@react-native-async-storage/async-storage';
import * as DocumentPicker from 'expo-document-picker';
import * as ImagePicker from 'expo-image-picker';
import { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';
// Actual Backend API - Change IP if running on a different device
const API_BASE_URL = 'http://192.168.68.125:5000'; // Flask backend IP

export default function UploadReportScreen({ navigation }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);

  const showImagePicker = () => {
    Alert.alert(
      'Select Image',
      'Choose an option',
      [
        { text: 'Camera', onPress: () => openCamera() },
        { text: 'Gallery', onPress: () => openGallery() },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  };

  const openCamera = async () => {
  const result = await ImagePicker.launchCameraAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.Images,
    quality: 0.8,
    allowsEditing: true,
  });

  if (!result.canceled) {
    setSelectedFile({
      uri: result.assets[0].uri,
      type: 'image/jpeg',
      name: 'camera_image.jpg',
      size: result.assets[0].fileSize || 0,
    });
  }
};

const openGallery = async () => {
  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.Images,
    quality: 0.8,
    allowsEditing: true,
  });

  if (!result.canceled) {
    setSelectedFile({
      uri: result.assets[0].uri,
      type: 'image/jpeg',
      name: 'gallery_image.jpg',
      size: result.assets[0].fileSize || 0,
    });
  }
};

  const selectDocument = async () => {
  try {
    const result = await DocumentPicker.getDocumentAsync({
      type: ['image/*', 'application/pdf'],
    });
    
    if (!result.canceled) {
      setSelectedFile({
        uri: result.assets[0].uri,
        type: result.assets[0].mimeType,
        name: result.assets[0].name,
        size: result.assets[0].size,
      });
    }
  } catch (error) {
    Alert.alert('Error', 'Failed to select document');
  }
};

  const uploadFile = async () => {
    if (!selectedFile) {
      Alert.alert('Error', 'Please select a file first');
      return;
    }

    setUploading(true);

    try {
      // Connect to the actual backend running at API_BASE_URL
      const mockResult = await apiCall(selectedFile);
      
      // Save to local storage for demo
      await saveReportLocally(mockResult);
      
      setResult(mockResult);
      
      // Navigate to result screen
      navigation.navigate('ReportResult', { result: mockResult });
      
    } catch (error) {
      Alert.alert('Error', 'Failed to upload file. Please try again.');
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }
  };

  // Real API call to the Flask backend
  const apiCall = async (file) => {
    try {
      // In a real scenario, we would send the FormData with the image/PDF
      // For this integration, the backend uses mock OCR when it gets hit on /analyze-report
      const response = await fetch(`${API_BASE_URL}/api/v1/analyze-report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: file.name })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      return result;
    } catch (error) {
      console.error("API Call failed:", error);
      throw error;
    }
  };

  const saveReportLocally = async (report) => {
    try {
      const existingReports = await AsyncStorage.getItem('userReports');
      const reports = existingReports ? JSON.parse(existingReports) : [];
      
      reports.unshift(report); // Add new report to beginning
      await AsyncStorage.setItem('userReports', JSON.stringify(reports));
    } catch (error) {
      console.error('Error saving report locally:', error);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setResult(null);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const isImage = (type) => {
    return type && type.startsWith('image/');
  };

  const isPDF = (type) => {
    return type && type === 'application/pdf';
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Upload Medical Report</Text>
          <Text style={styles.subtitle}>
            Select an image or PDF of your medical report for AI analysis
          </Text>
        </View>

        {/* Upload Options */}
        {!selectedFile && (
          <View style={styles.uploadOptions}>
            <TouchableOpacity style={styles.uploadOption} onPress={showImagePicker}>
              <View style={[styles.uploadIconContainer, { backgroundColor: '#4CAF50' + '20' }]}>
                <Icon name="camera" size={32} color="#4CAF50" />
              </View>
              <Text style={styles.uploadOptionTitle}>Camera/Gallery</Text>
              <Text style={styles.uploadOptionSubtitle}>Take photo or select from gallery</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.uploadOption} onPress={selectDocument}>
              <View style={[styles.uploadIconContainer, { backgroundColor: '#2196F3' + '20' }]}>
                <Icon name="document-text" size={32} color="#2196F3" />
              </View>
              <Text style={styles.uploadOptionTitle}>Documents</Text>
              <Text style={styles.uploadOptionSubtitle}>Select PDF or image files</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Selected File Preview */}
        {selectedFile && (
          <View style={styles.filePreview}>
            <View style={styles.filePreviewHeader}>
              <Text style={styles.filePreviewTitle}>Selected File</Text>
              <TouchableOpacity onPress={removeFile} style={styles.removeButton}>
                <Icon name="close-circle" size={24} color="#F44336" />
              </TouchableOpacity>
            </View>

            <View style={styles.fileInfo}>
              {isImage(selectedFile.type) && (
                <Image source={{ uri: selectedFile.uri }} style={styles.imagePreview} />
              )}
              
              {isPDF(selectedFile.type) && (
                <View style={styles.pdfPreview}>
                  <Icon name="document-text" size={48} color="#F44336" />
                  <Text style={styles.pdfText}>PDF Document</Text>
                </View>
              )}

              <View style={styles.fileDetails}>
                <Text style={styles.fileName}>{selectedFile.name}</Text>
                <Text style={styles.fileSize}>{formatFileSize(selectedFile.size)}</Text>
              </View>
            </View>

            <TouchableOpacity
              style={[styles.uploadButton, uploading && styles.uploadButtonDisabled]}
              onPress={uploadFile}
              disabled={uploading}
            >
              {uploading ? (
                <View style={styles.uploadingContainer}>
                  <ActivityIndicator size="small" color="white" />
                  <Text style={styles.uploadButtonText}>Processing...</Text>
                </View>
              ) : (
                <>
                  <Icon name="cloud-upload" size={20} color="white" />
                  <Text style={styles.uploadButtonText}>Analyze Report</Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        )}

        {/* Upload Guidelines */}
        <View style={styles.guidelines}>
          <Text style={styles.guidelinesTitle}>Upload Guidelines</Text>
          
          <View style={styles.guideline}>
            <Icon name="checkmark-circle" size={16} color="#4CAF50" />
            <Text style={styles.guidelineText}>Ensure report is clearly visible and readable</Text>
          </View>
          
          <View style={styles.guideline}>
            <Icon name="checkmark-circle" size={16} color="#4CAF50" />
            <Text style={styles.guidelineText}>Supported formats: JPG, PNG, PDF</Text>
          </View>
          
          <View style={styles.guideline}>
            <Icon name="checkmark-circle" size={16} color="#4CAF50" />
            <Text style={styles.guidelineText}>Maximum file size: 10MB</Text>
          </View>
          
          <View style={styles.guideline}>
            <Icon name="information-circle" size={16} color="#2196F3" />
            <Text style={styles.guidelineText}>Your data is processed securely and confidentially</Text>
          </View>
        </View>

        {/* Recent Uploads */}
        <View style={styles.recentSection}>
          <Text style={styles.recentTitle}>Recent Analysis</Text>
          <Text style={styles.recentSubtitle}>
            Previous reports are saved in your profile for easy access
          </Text>
          
          <TouchableOpacity 
            style={styles.viewHistoryButton}
            onPress={() => navigation.navigate('Home')}
          >
            <Icon name="time" size={20} color="#2196F3" />
            <Text style={styles.viewHistoryText}>View History</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollView: {
    flex: 1,
  },
  header: {
    padding: 20,
    backgroundColor: 'white',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginTop: 8,
    lineHeight: 22,
  },
  uploadOptions: {
    padding: 20,
  },
  uploadOption: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    marginBottom: 15,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  uploadIconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  uploadOptionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  uploadOptionSubtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  filePreview: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  filePreviewHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  filePreviewTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  removeButton: {
    padding: 4,
  },
  fileInfo: {
    alignItems: 'center',
    marginBottom: 20,
  },
  imagePreview: {
    width: 200,
    height: 200,
    borderRadius: 8,
    marginBottom: 12,
  },
  pdfPreview: {
    alignItems: 'center',
    padding: 20,
    marginBottom: 12,
  },
  pdfText: {
    fontSize: 14,
    color: '#666',
    marginTop: 8,
  },
  fileDetails: {
    alignItems: 'center',
  },
  fileName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
  },
  fileSize: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  uploadButton: {
    backgroundColor: '#2196F3',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 15,
    borderRadius: 10,
  },
  uploadButtonDisabled: {
    opacity: 0.6,
  },
  uploadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  uploadButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  guidelines: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  guidelinesTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 15,
  },
  guideline: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  guidelineText: {
    fontSize: 14,
    color: '#555',
    marginLeft: 8,
    flex: 1,
  },
  recentSection: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  recentTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  recentSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
    marginBottom: 15,
  },
  viewHistoryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    backgroundColor: '#2196F3' + '15',
    borderRadius: 8,
  },
  viewHistoryText: {
    color: '#2196F3',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 8,
  },
});