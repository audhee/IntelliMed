import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Icon from 'react-native-vector-icons/Ionicons';

// Import screens
import LoginScreen from './screens/LoginScreen';
import PatientHomeScreen from './screens/PatientHomeScreen';
import DoctorHomeScreen from './screens/DoctorHomeScreen';
import UploadReportScreen from './screens/UploadReportScreen';
import ReportResultScreen from './screens/ReportResultScreen';
import ChatbotScreen from './screens/ChatbotScreen';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

// Patient Tab Navigator
function PatientTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;
          
          if (route.name === 'Home') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'Upload') {
            iconName = focused ? 'cloud-upload' : 'cloud-upload-outline';
          } else if (route.name === 'Chat') {
            iconName = focused ? 'chatbubble' : 'chatbubble-outline';
          }
          
          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#2196F3',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
      })}
    >
      <Tab.Screen name="Home" component={PatientHomeScreen} />
      <Tab.Screen name="Upload" component={UploadReportScreen} />
      <Tab.Screen name="Chat" component={ChatbotScreen} />
    </Tab.Navigator>
  );
}

// Doctor Tab Navigator
function DoctorTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;
          
          if (route.name === 'Home') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'Reports') {
            iconName = focused ? 'document-text' : 'document-text-outline';
          }
          
          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#4CAF50',
        tabBarInactiveTintColor: 'gray',
        headerShown: false,
      })}
    >
      <Tab.Screen name="Home" component={DoctorHomeScreen} />
      <Tab.Screen name="Reports" component={DoctorHomeScreen} />
    </Tab.Navigator>
  );
}

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userRole, setUserRole] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
  try {
    const token = await AsyncStorage.getItem('userToken');
    const role = await AsyncStorage.getItem('userRole');
    
    if (token && role) {
      setIsLoggedIn(true);
      setUserRole(role);
    } else {
      // Important: Reset state when no valid auth found
      setIsLoggedIn(false);
      setUserRole(null);
    }
  } catch (error) {
    console.error('Error checking auth status:', error);
    setIsLoggedIn(false);
    setUserRole(null);
  } finally {
    setLoading(false);
  }
};

// Add this useEffect to check auth status periodically
useEffect(() => {
  checkAuthStatus();
  
  // Check every 2 seconds for auth changes
  const interval = setInterval(checkAuthStatus, 2000);
  
  return () => clearInterval(interval);
}, []);

  if (loading) {
    return null; // You can add a loading screen here
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {!isLoggedIn ? (
          <Stack.Screen name="Login">
            {(props) => (
              <LoginScreen
                {...props}
                onLogin={(role) => {
                  setIsLoggedIn(true);
                  setUserRole(role);
                }}
              />
            )}
          </Stack.Screen>
        ) : (
          <>
            {userRole === 'patient' ? (
              <Stack.Screen name="PatientApp" component={PatientTabs} />
            ) : (
              <Stack.Screen name="DoctorApp" component={DoctorTabs} />
            )}
            <Stack.Screen 
              name="ReportResult" 
              component={ReportResultScreen}
              options={{ headerShown: true, title: 'Report Results' }}
            />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}

// Register with Expo
import { registerRootComponent } from 'expo';
registerRootComponent(App);