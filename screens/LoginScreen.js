import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Icon from 'react-native-vector-icons/Ionicons';

// API Connection IP - Enters port 5000 for local FastAPI
const API_BASE_URL = 'http://192.168.68.120:5000';

const LoginScreen = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    setLoading(true);

    const lowerEmail = email.trim().toLowerCase();
    
    // Offline local support: Short-circuit database check for Demo accounts so they always work!
    if (lowerEmail === 'patient@test.com' && password === '123456') {
      try {
        await AsyncStorage.setItem('userToken', 'demo-patient-token');
        await AsyncStorage.setItem('userRole', 'patient');
        await AsyncStorage.setItem('userName', 'Demo Patient');
        await AsyncStorage.setItem('userEmail', 'patient@test.com');
        
        // Seed beautiful, highly detailed medical reports for the offline dashboard demo!
        const demoReports = [
          {
            id: "report-1",
            title: "Complete Blood Count (CBC) Panel",
            date: "5/28/2026",
            status: "Analyzed",
            confidence: 0.94,
            diagnosis: "Mild iron deficiency detected with low Hemoglobin (11.2 g/dL) and borderline low Red Blood Cell count. Other parameters (WBC, Platelets) are within healthy canonical references.",
            prescription: "Ferrous sulfate 325mg daily, combined with daily dietary alterations. Re-test hemoglobin parameters in 60 days.",
            recommendations: [
              "Increase dietary iron intake (spinach, lentils, clean poultry/red meat).",
              "Pair non-heme iron meals with Vitamin C (e.g. lemon juice) to improve gastrointestinal absorption.",
              "Avoid consuming coffee, black tea, or calcium supplements within 1 hour of iron-rich meals.",
              "Track daily energy scores and monitor for symptoms like fatigue or dizziness."
            ]
          },
          {
            id: "report-2",
            title: "Metabolic & Glycemic Profile",
            date: "5/10/2026",
            status: "Analyzed",
            confidence: 0.89,
            diagnosis: "Slightly elevated Fasting Plasma Glucose (104 mg/dL), indicating borderline impaired fasting glucose (prediabetic reference). Thyroid (TSH) and Lipid panel parameters remain fully stable.",
            prescription: "Lifestyle modification and nutritional counseling. No active pharmaceutical prescription indicated at this stage.",
            recommendations: [
              "Incorporate regular aerobic cardiovascular exercise (at least 30 minutes, 5 days a week).",
              "Transition toward a low-glycemic dietary model, emphasizing fiber, slow carbs, and clean proteins.",
              "Limit consumption of processed sugars, simple flours, and sweetened beverages.",
              "Obtain regular fasting metabolic panels every 6 months to monitor glycemic trends."
            ]
          }
        ];
        
        await AsyncStorage.setItem('recentReports', JSON.stringify(demoReports));
        await AsyncStorage.setItem('userReports', JSON.stringify(demoReports));

        onLogin('patient');
        setLoading(false);
        return;
      } catch (error) {
        console.error('Error saving local demo session:', error);
      }
    } else if (lowerEmail === 'doctor@test.com' && password === '123456') {
      try {
        await AsyncStorage.setItem('userToken', 'demo-doctor-token');
        await AsyncStorage.setItem('userRole', 'doctor');
        await AsyncStorage.setItem('userName', 'Demo Doctor');
        await AsyncStorage.setItem('userEmail', 'doctor@test.com');
        onLogin('doctor');
        setLoading(false);
        return;
      } catch (error) {
        console.error('Error saving local demo session:', error);
      }
    }

    // Normal backend route for real accounts
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: email.trim(),
          password: password,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Store real JWT session credentials returned by FastAPI
        await AsyncStorage.setItem('userToken', data.access_token);
        await AsyncStorage.setItem('userRole', data.role);
        await AsyncStorage.setItem('userName', data.full_name);
        await AsyncStorage.setItem('userEmail', email.trim());
        
        // Notify parent application state
        onLogin(data.role);
      } else {
        const errorMsg = data.detail || 'Invalid email or password.';
        Alert.alert('Login Failed', errorMsg);
      }
    } catch (error) {
      console.error('Login error details:', error);
      Alert.alert('Connection Error', 'Failed to reach health server. Verify your local IP and uvicorn running state.');
    } finally {
      setLoading(false);
    }
  };

  const fillDemoCredentials = (type) => {
    if (type === 'patient') {
      setEmail('patient@test.com');
      setPassword('123456'); // Updated to meet min length of 6
    } else {
      setEmail('doctor@test.com');
      setPassword('123456');
    }
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <View style={styles.header}>
        <Image 
          source={require('../assets/logo.png')} 
          style={styles.logo} 
          resizeMode="contain" 
        />
      </View>

      <View style={styles.form}>
        <View style={styles.inputContainer}>
          <Icon name="mail-outline" size={20} color="#123C58" style={styles.inputIcon} />
          <TextInput
            style={styles.input}
            placeholder="Email"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
          />
        </View>

        <View style={styles.inputContainer}>
          <Icon name="lock-closed-outline" size={20} color="#123C58" style={styles.inputIcon} />
          <TextInput
            style={styles.input}
            placeholder="Password"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />
        </View>

        <TouchableOpacity 
          style={styles.loginButton} 
          onPress={handleLogin}
          disabled={loading}
        >
          <Text style={styles.loginButtonText}>
            {loading ? 'Logging in...' : 'Login'}
          </Text>
        </TouchableOpacity>

        <View style={styles.demoSection}>
          <Text style={styles.demoTitle}>Try Demo Accounts:</Text>
          <View style={styles.demoButtons}>
            <TouchableOpacity 
              style={[styles.demoButton, styles.patientDemo]}
              onPress={() => fillDemoCredentials('patient')}
            >
              <Icon name="person" size={16} color="#fff" />
              <Text style={styles.demoButtonText}>Patient Demo</Text>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.demoButton, styles.doctorDemo]}
              onPress={() => fillDemoCredentials('doctor')}
            >
              <Icon name="medical" size={16} color="#fff" />
              <Text style={styles.demoButtonText}>Doctor Demo</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    justifyContent: 'center',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 30,
    marginTop: 40,
    width: '100%',
  },
  logo: {
    width: 320,
    height: 100,
    alignSelf: 'center',
  },
  form: {
    backgroundColor: '#fff',
    padding: 30,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#eee',
    borderRadius: 10,
    marginBottom: 15,
    paddingHorizontal: 15,
    backgroundColor: '#f8f9fa',
  },
  inputIcon: {
    marginRight: 10,
  },
  input: {
    flex: 1,
    paddingVertical: 15,
    fontSize: 16,
    color: '#333',
  },
  loginButton: {
    backgroundColor: '#123C58',
    paddingVertical: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 10,
  },
  loginButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  demoSection: {
    marginTop: 30,
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  demoTitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 15,
  },
  demoButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  demoButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 8,
    flex: 0.48,
    justifyContent: 'center',
  },
  patientDemo: {
    backgroundColor: '#158C86',
  },
  doctorDemo: {
    backgroundColor: '#2F5D7C',
  },
  demoButtonText: {
    color: '#fff',
    fontSize: 12,
    marginLeft: 5,
    fontWeight: '600',
  },
});

export default LoginScreen;