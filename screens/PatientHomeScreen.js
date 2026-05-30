import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Alert,
  Image,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Icon from 'react-native-vector-icons/Ionicons';

// API Connection IP - Enters port 5000 for local FastAPI
const API_BASE_URL = 'http://192.168.68.120:5000';

export default function PatientHomeScreen({ navigation }) {
  const [patientName, setPatientName] = useState('Patient');
  const [recentReports, setRecentReports] = useState([]);

  // Auto-fetch whenever the dashboard screen comes into active focus!
  useEffect(() => {
    const unsubscribe = navigation.addListener('focus', () => {
      loadPatientData();
    });
    loadPatientData(); // Initial load
    return unsubscribe;
  }, [navigation]);

  const loadPatientData = async () => {
    try {
      const name = await AsyncStorage.getItem('userName');
      if (name) {
        setPatientName(name);
      }
      
      const token = await AsyncStorage.getItem('userToken');
      if (!token) return;

      // Fetch dynamic analysis history from Supabase main DB via FastAPI
      const response = await fetch(`${API_BASE_URL}/api/v1/reports/history?page=1&limit=5`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Accept': 'application/json',
        },
      });

      if (response.ok) {
        const reports = await response.json();
        
        // Transform backend keys if necessary to match Patient HomeScreen schema
        const mappedReports = reports.map(r => ({
          id: r.id,
          title: r.filename || 'Analyzed Report',
          date: new Date(r.timestamp).toLocaleDateString(),
          status: 'Analyzed',
          confidence: r.confidence,
          diagnosis: r.diagnosis,
          prescription: r.prescription,
          recommendations: r.recommendations
        }));

        setRecentReports(mappedReports);
        await AsyncStorage.setItem('recentReports', JSON.stringify(mappedReports));
      } else {
        // Fallback to offline cached storage on connection failure
        const reports = await AsyncStorage.getItem('recentReports');
        if (reports) {
          setRecentReports(JSON.parse(reports));
        }
      }
    } catch (error) {
      console.warn('Error loading patient dynamic feed (using offline fallback):', error.message || error);
      // Offline fallback
      const reports = await AsyncStorage.getItem('recentReports');
      if (reports) {
        setRecentReports(JSON.parse(reports));
      }
    }
  };

  const handleLogout = async () => {
  Alert.alert(
    'Logout',
    'Are you sure you want to logout?',
    [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Logout',
        onPress: async () => {
          try {
            await AsyncStorage.multiRemove(['userToken', 'userRole', 'userName']);
            // App.js periodic auth check will automatically detect token deletion and redirect to Login screen
          } catch (error) {
            console.error('Error during logout:', error);
          }
        },
      },
    ]
  );
};

  const QuickActionCard = ({ title, subtitle, iconName, onPress, color }) => (
    <TouchableOpacity style={[styles.actionCard, { borderLeftColor: color }]} onPress={onPress}>
      <View style={styles.actionCardContent}>
        <View style={styles.actionIcon}>
          <Icon name={iconName} size={28} color={color} />
        </View>
        <View style={styles.actionText}>
          <Text style={styles.actionTitle}>{title}</Text>
          <Text style={styles.actionSubtitle}>{subtitle}</Text>
        </View>
        <Icon name="chevron-forward" size={20} color="#ccc" />
      </View>
    </TouchableOpacity>
  );

  const RecentReportCard = ({ report }) => (
    <TouchableOpacity 
      style={styles.reportCard}
      onPress={() => navigation.navigate('ReportResult', { 
        result: {
          id: report.id,
          filename: report.title,
          timestamp: report.date,
          confidence: report.confidence,
          diagnosis: report.diagnosis,
          prescription: report.prescription,
          recommendations: report.recommendations
        } 
      })}
    >
      <View style={styles.reportHeader}>
        <Icon name="document-text" size={20} color="#2196F3" />
        <Text style={styles.reportTitle}>{report.title}</Text>
      </View>
      <Text style={styles.reportDate}>{report.date}</Text>
      <Text style={styles.reportStatus}>{report.status}</Text>
    </TouchableOpacity>
  );

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLogoContainer}>
          <Image 
            source={require('../assets/logo.png')} 
            style={styles.headerLogo} 
            resizeMode="contain" 
          />
        </View>
        <TouchableOpacity onPress={handleLogout} style={styles.logoutButton}>
          <Icon name="log-out-outline" size={24} color="#123C58" />
        </TouchableOpacity>
      </View>

      <View style={styles.welcomeSection}>
        <Text style={styles.greeting}>Good Morning,</Text>
        <Text style={styles.patientName}>{patientName}</Text>
      </View>

      {/* Health Summary Card */}
      <View style={styles.summaryCard}>
        <View style={styles.summaryHeader}>
          <Icon name="shield-checkmark" size={24} color="#158C86" />
          <Text style={styles.summaryTitle}>Health Summary</Text>
        </View>
        <Text style={styles.summaryText}>
          Your health is our priority. IntelliMed delivers AI-Powered Longitudinal Health Intelligence to guide you step-by-step.
        </Text>
      </View>

      {/* Quick Actions */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        
        <QuickActionCard
          title="Upload Report"
          subtitle="Get AI analysis of your medical reports"
          iconName="cloud-upload"
          color="#123C58"
          onPress={() => navigation.navigate('Upload')}
        />
        
        <QuickActionCard
          title="Health Chatbot"
          subtitle="Ask questions about your health"
          iconName="chatbubble-ellipses"
          color="#158C86"
          onPress={() => navigation.navigate('Chat')}
        />
        
        <QuickActionCard
          title="Emergency Info"
          subtitle="Important emergency contacts"
          iconName="medical"
          color="#FF5722"
          onPress={() => Alert.alert('Emergency', 'Call 108 for medical emergency')}
        />
      </View>

      {/* Recent Reports */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Recent Reports</Text>
        {recentReports.length > 0 ? (
          recentReports.map((report, index) => (
            <RecentReportCard key={index} report={report} />
          ))
        ) : (
          <View style={styles.emptyState}>
            <Icon name="document-outline" size={48} color="#ccc" />
            <Text style={styles.emptyText}>No reports uploaded yet</Text>
            <Text style={styles.emptySubtext}>Upload your first report to get started</Text>
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 45,
    paddingBottom: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  headerLogoContainer: {
    width: 140,
    height: 40,
  },
  headerLogo: {
    width: '100%',
    height: '100%',
  },
  welcomeSection: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 10,
  },
  greeting: {
    fontSize: 15,
    color: '#666',
  },
  patientName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#123C58',
    marginTop: 4,
  },
  logoutButton: {
    padding: 8,
  },
  summaryCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginVertical: 10,
    padding: 20,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#158C86',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
  },
  summaryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#123C58',
    marginLeft: 12,
  },
  summaryText: {
    fontSize: 14,
    color: '#555',
    lineHeight: 22,
  },
  section: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  actionCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    borderLeftWidth: 4,
  },
  actionCardContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  actionIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#f8f9fa',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  actionText: {
    flex: 1,
  },
  actionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  actionSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  reportCard: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  reportHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  reportTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 8,
  },
  reportDate: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  reportStatus: {
    fontSize: 14,
    color: '#4CAF50',
    fontWeight: '500',
  },
  emptyState: {
    alignItems: 'center',
    padding: 40,
  },
  emptyText: {
    fontSize: 18,
    color: '#666',
    marginTop: 16,
    fontWeight: '500',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 8,
    textAlign: 'center',
  },
});