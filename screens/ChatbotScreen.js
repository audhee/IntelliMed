import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Image,
} from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';
import AsyncStorage from '@react-native-async-storage/async-storage';

// API Connection IP - Enters port 5000 for local FastAPI
const API_BASE_URL = 'http://192.168.68.120:5000';

export default function ChatbotScreen({ route }) {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [reportContext, setReportContext] = useState(null);
  const flatListRef = useRef(null);

  useEffect(() => {
    // Initial greeting message
    const initialMessage = {
      id: '1',
      text: 'Hello! I\'m your IntelliMed AI Health Assistant, powered by our Longitudinal Health Intelligence. I can help you interpret medical reports, explain clinical findings, advise on personalized dietary support, or answer fitness and medication safety questions. How can I assist you today?',
      isBot: true,
      timestamp: new Date().toISOString(),
    };
    setMessages([initialMessage]);
  }, []);

  // Capture report context from parameters
  useEffect(() => {
    if (route?.params?.reportContext) {
      const context = route.params.reportContext;
      setReportContext(context);
      
      const contextMsg = {
        id: 'report_context_loaded',
        text: `📊 **Medical Report Context Loaded**:\n• **File**: ${context.filename || 'blood_report.jpg'}\n• **Findings**: ${context.diagnosis}\n\n*I am now primed with your lab report details. Ask me any questions about your results, diet recommendations, or next steps!*`,
        isBot: true,
        timestamp: new Date().toISOString(),
      };
      
      setMessages(prev => {
        if (prev.some(m => m.id === 'report_context_loaded')) {
          return prev.map(m => m.id === 'report_context_loaded' ? contextMsg : m);
        }
        return [...prev, contextMsg];
      });
    }
  }, [route?.params?.reportContext]);

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputText.trim(),
      isBot: false,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    // Simulate API call to backend chatbot
    try {
      // Replace this with actual API call
      const botResponse = await simulateBotResponse(userMessage.text);
      
      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: botResponse,
        isBot: true,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error. Please try again.',
        isBot: true,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  // FIXED: Enhanced Gemini API integration with live FastAPI connection
  const simulateBotResponse = async (userInput) => {
    try {
      const token = await AsyncStorage.getItem('userToken');
      if (!token) throw new Error("No active JWT session token found.");

      // 1. Send live POST query to FastAPI Gemini Chat router
      const response = await fetch(`${API_BASE_URL}/api/v1/reports/chat`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          query: userInput,
          report_context: reportContext, // Pass loaded report findings dynamically!
        }),
      });

      const data = await response.json();
      if (response.ok && data.response) {
        return data.response; // Return genuine live Gemini-generated advice!
      } else {
        throw new Error(data.detail || "FastAPI Gemini processing failure.");
      }
    } catch (error) {
      console.warn("FastAPI chatbot down, falling back to clinical rules:", error);
    }

    // ----------------------------------------------------
    // CLINICAL RESILIENT LOCAL FALLBACK (Offline Support)
    // ----------------------------------------------------
    const input = userInput.toLowerCase().trim();
    
    // Dynamic report-specific local fallback if uvicorn is offline
    if (reportContext) {
      const lowerDiagnosis = (reportContext.diagnosis || '').toLowerCase();
      const lowerPrescription = (reportContext.prescription || '').toLowerCase();
      const lowerRecs = (reportContext.recommendations || []).join(' ').toLowerCase();
      const combinedContextText = `${lowerDiagnosis} ${lowerPrescription} ${lowerRecs}`;

      // 1. If asking about report, explain diagnosis, etc.
      if (
        input.includes('my report') || 
        input.includes('my result') || 
        input.includes('explain') || 
        input.includes('diagnosis') || 
        input.includes('prescription') ||
        input.includes('findings') ||
        input.includes('what should i do')
      ) {
        const recs = reportContext.recommendations || [];
        const recList = recs.length > 0 
          ? recs.map((rec, i) => `• ${rec}`).join('\n')
          : '• Monitor your daily dietary intake.\n• Maintain adequate hydration levels.';
          
        return `📋 **Your Lab Report Advisory Context**:\n\n**AI Diagnosis Summary**:\n${reportContext.diagnosis || 'Standard blood panel results.'}\n\n**Suggested Treatment / Advices**:\n${reportContext.prescription || 'Lifestyle monitoring.'}\n\n**Actionable Recommendations**:\n${recList}\n\n*Remember, these summaries represent AI findings. Always consult your primary care doctor to discuss these metrics.*`;
      }

      // 2. If asking about eating/diet/food/nutrition with active report context
      if (
        input.includes('diet') || 
        input.includes('nutrition') || 
        input.includes('food') || 
        input.includes('eat') || 
        input.includes('eating') ||
        input.includes('drink')
      ) {
        // Glycemic / Metabolic issues
        if (
          combinedContextText.includes('glucose') || 
          combinedContextText.includes('sugar') || 
          combinedContextText.includes('diabetic') || 
          combinedContextText.includes('glycemic')
        ) {
          return `🥦 **Recommended Diet Plan (Metabolic/Glycemic Support)**:\n\nBased on your elevated Fasting Glucose metrics in your active report, we recommend a low-glycemic eating pattern to support healthy metabolic responses:\n\n• **Include**: Abundant leafy greens (spinach, broccoli, kale), slow-digesting fibers (chia seeds, oats, quinoa), and clean proteins (lentils, fish, chicken, eggs).\n• **Avoid**: Refined flours, added processed sugars, sweet carbonated beverages, and simple white breads.\n• **Tip**: Pair carbohydrates with fiber and healthy fats to minimize post-prandial blood sugar spikes, and record your daily energy indexes!`;
        }

        // Vitamin D deficiency
        if (
          combinedContextText.includes('vitamin d') || 
          combinedContextText.includes('vit d') || 
          combinedContextText.includes('d3') || 
          combinedContextText.includes('calcium')
        ) {
          return `☀️ **Recommended Diet Plan (Vitamin D & Bone Density Support)**:\n\nBased on your lower Vitamin D scores in your active report, we recommend focusing on vitamin-dense foods and safe sunlight absorption:\n\n• **Include**: Fat-soluble foods (fatty fish like salmon/mackerel, egg yolks), calcium-reinforced milk or plant alternatives, and fortified mushrooms.\n• **Outdoor lifestyle**: Target 10-15 minutes of safe midday sunlight daily, which prompts natural synthesis.\n• **Tip**: Vitamin D3 works in synergy with Vitamin K2 and Magnesium, so incorporate seeds, nuts, and leafy greens too!`;
        }

        // Anemia / Low Hemoglobin / Iron
        if (
          combinedContextText.includes('hemoglobin') || 
          combinedContextText.includes('iron') || 
          combinedContextText.includes('anemi') || 
          combinedContextText.includes('rbc') || 
          combinedContextText.includes('red blood')
        ) {
          return `🥩 **Recommended Diet Plan (Iron & Red Blood Cell Support)**:\n\nBased on your red blood cell indicators (e.g. Hemoglobin), we recommend focusing on hematinic foods to optimize energy baselines:\n\n• **Heme Iron (High bioavailability)**: Lean beef, chicken, or shellfish.\n• **Non-Heme Iron (Plant-based)**: Organic spinach, kidney beans, lentils, pumpkin seeds.\n• **Crucial tip**: Pair non-heme iron with Vitamin C (e.g. squeeze lemon juice on your salad) to dramatically increase absorption, and avoid coffee/tea within 1 hour of meals!`;
        }

        // Default customized diet using active report context
        return `🥗 **Report-Tailored Nutritional Guidelines**:\n\nLooking at your active medical report summary ("*${reportContext.diagnosis || 'Lifestyle monitoring panel'}*"), we recommend a balanced anti-inflammatory nutritional pattern:\n\n• **50% of your plate**: Color-rich leafy vegetables, cruciferous greens, and low-fructose berries.\n• **25% of your plate**: Cellular-building lean proteins (e.g. legumes, wild-caught fish, organic poultry).\n• **25% of your plate**: High-fiber complex whole grains (e.g. brown rice, oats, barley).\n• **Actionable advice**: Review the recommended steps in your report card:\n${(reportContext.recommendations || []).slice(0, 2).map(r => `  • ${r}`).join('\n') || '  • Maintain regular physical activity and baseline metrics.'}\n\n*Consult a registered dietitian or your physician before making significant dietary modifications.*`;
      }
    }

    // 3. Fallback when there is no report context
    if (
      input.includes('diet') || 
      input.includes('nutrition') || 
      input.includes('food') || 
      input.includes('eat') || 
      input.includes('eating') ||
      input.includes('drink')
    ) {
      return `🥗 **General Healthy Nutrition Guidelines**:

To promote long-term vitality, focus on a balanced Mediterranean-style eating pattern:

• **50% plate**: Color-rich leafy vegetables and fresh low-fructose fruits.
• **25% plate**: Lean, cell-building proteins (e.g. fish, poultry, legumes).
• **25% plate**: Complex high-fiber whole grains (e.g. brown rice, oats).
• **Hydration**: Aim for 8-10 glasses of pure water daily to maintain metabolic pathways.
• **Avoid**: Processed and packaged foods, refined simple sugars, carbonated beverages, trans fats, and excess sodium.

*Consult a registered dietitian or your physician for personalized dietary plans tailored to your specific metabolic baselines.*`;
    }
  
  // Exercise and fitness
  if (input.includes('exercise') || input.includes('workout') || input.includes('fitness') || input.includes('physical activity')) {
    return `Exercise guidelines:

🏃 **Start slowly** - especially if you're new to exercise
⏰ **150 minutes** moderate exercise per week (WHO recommendation)

💪 **Include both:**
• **Cardio:** walking, swimming, cycling
• **Strength:** bodyweight exercises, weights

🧘 **Don't forget:**
• Flexibility and stretching
• Warm-up and cool-down
• Rest days for recovery

⚠️ **Safety tips:**
• Stay hydrated
• Listen to your body
• Stop if you feel pain
• Start with 10-15 minutes daily

**Consult a doctor before starting if you:**
• Have chronic conditions
• Are over 40 and sedentary
• Have heart conditions
• Take medications

*A fitness professional can create personalized workout plans.*`;
  }
  
  // General health symptoms
  if (input.includes('pain') || input.includes('sick') || input.includes('unwell')) {
    return `For general health concerns:

🩺 **When to see a doctor:**
• Symptoms persist or worsen
• High fever or severe pain
• Difficulty breathing
• Chest pain or pressure
• Severe abdominal pain
• Sudden vision or speech changes

🏠 **Self-care basics:**
• Rest and adequate sleep
• Stay hydrated
• Eat nutritious foods
• Avoid stress when possible

📞 **Emergency contacts:**
• Emergency services: 108
• Poison control if needed
• Your regular doctor

*I provide general information only. For specific symptoms or health concerns, always consult qualified healthcare professionals for proper diagnosis and treatment.*`;
  }
  
  // Medication queries
  if (input.includes('medicine') || input.includes('medication') || input.includes('drug') || input.includes('tablet')) {
    return `Medication safety:

💊 **Always:**
• Follow prescribed dosages exactly
• Take at recommended times
• Complete full courses (antibiotics)
• Store medications properly
• Check expiration dates

⚠️ **Never:**
• Share prescription medications
• Exceed recommended doses
• Mix medications without doctor approval
• Stop prescribed medications suddenly

🤔 **Questions to ask your doctor:**
• How and when to take medication
• Possible side effects
• Food/drink interactions
• Other medication interactions

📞 **Contact doctor if:**
• Side effects occur
• Symptoms don't improve
• You miss doses
• You have concerns

*Only qualified healthcare providers can prescribe and advise on medications. Never self-medicate.*`;
  }
  
  // Mental health
  if (input.includes('stress') || input.includes('anxiety') || input.includes('depression') || input.includes('mental') || input.includes('mood')) {
    return `Mental health support:

🧠 **Stress management:**
• Deep breathing exercises
• Regular physical activity
• Adequate sleep (7-9 hours)
• Connect with friends and family
• Practice mindfulness or meditation

📞 **Professional help:**
• Talk to your primary care doctor
• Consider counseling or therapy
• Mental health helplines available
• Don't hesitate to seek support

⚠️ **Seek immediate help if:**
• Thoughts of self-harm
• Severe depression or anxiety
• Unable to function daily
• Substance use concerns

🌟 **Remember:**
• Mental health is as important as physical health
• Seeking help is a sign of strength
• Treatment is effective and available

*Mental health professionals can provide proper assessment and treatment. Don't suffer in silence.*`;
  }
  
  // Default response for other queries
  return `Thank you for your health question: "${userInput}"

🩺 **For specific health concerns, I recommend:**
• Consulting with a qualified healthcare professional
• Getting proper medical examination
• Following professional medical advice
• Keeping a symptom diary if ongoing

📞 **Emergency contacts:**
• Emergency services: 108
• Your family doctor
• Local hospital or clinic

💡 **I can help with general information about:**
• Common symptoms (fever, headache)
• Healthy lifestyle tips
• When to seek medical care
• Basic first aid guidance

*I provide general health information only and cannot replace professional medical advice, diagnosis, or treatment.*`;
};

// You can also add this helper function for more interactive responses
const getHealthTip = () => {
  const tips = [
    "Drink at least 8 glasses of water daily for optimal health.",
    "Aim for 7-9 hours of quality sleep each night.",
    "Include at least 30 minutes of physical activity in your day.",
    "Eat a rainbow of fruits and vegetables for diverse nutrients.",
    "Practice stress management techniques like deep breathing.",
    "Wash your hands regularly to prevent infections.",
    "Take breaks from screens to rest your eyes.",
    "Maintain good posture, especially when sitting long hours."
  ];
  
  return tips[Math.floor(Math.random() * tips.length)];
};

// Fallback function with basic health responses
const getHealthcareResponse = (userInput) => {
  const input = userInput.toLowerCase();
  
  if (input.includes('fever')) {
    return `For fever management:
• Rest and stay hydrated
• Take temperature regularly
• Use fever reducers as directed
• See a doctor if fever persists or is very high
• Seek immediate care if accompanied by severe symptoms

Please consult a healthcare provider for proper diagnosis.`;
  }
  
  if (input.includes('headache')) {
    return `For headache relief:
• Rest in a quiet, dark room
• Stay hydrated
• Apply cold/warm compress
• Gentle neck/shoulder massage
• Over-the-counter pain relievers as directed

Consult a doctor if headaches are severe or frequent.`;
  }
  
  if (input.includes('diet') || input.includes('nutrition')) {
    return `For healthy diet tips:
• Eat plenty of fruits and vegetables
• Choose whole grains
• Include lean proteins
• Stay hydrated
• Limit processed foods

Consult a nutritionist for personalized diet plans.`;
  }
  
  if (input.includes('exercise') || input.includes('fitness')) {
    return `For exercise advice:
• Start slowly and gradually increase intensity
• Include cardio and strength training
• Stay hydrated during workouts
• Get adequate rest between sessions
• Listen to your body

Consult a fitness professional or doctor before starting new routines.`;
  }
  
  return `Thank you for your question about "${userInput}". 

For health-related concerns, I recommend:
• Consulting with a qualified healthcare professional
• Getting proper medical examination and diagnosis
• Following professional medical advice
• For emergencies, call 108 or visit nearest hospital

I can provide general health information, but cannot replace professional medical advice.`;
  };
  const renderMessage = ({ item }) => (
    <View style={[
      styles.messageContainer,
      item.isBot ? styles.botMessage : styles.userMessage
    ]}>
      {item.isBot && (
        <View style={styles.botAvatar}>
          <Icon name="shield-checkmark" size={16} color="#fff" />
        </View>
      )}
      <View style={[
        styles.messageBubble,
        item.isBot ? styles.botBubble : styles.userBubble
      ]}>
        <Text style={[
          styles.messageText,
          item.isBot ? styles.botText : styles.userText
        ]}>
          {item.text}
        </Text>
        <Text style={[
          styles.timestamp,
          item.isBot ? styles.botTimestamp : styles.userTimestamp
        ]}>
          {new Date(item.timestamp).toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
          })}
        </Text>
      </View>
    </View>
  );

  const QuickReply = ({ text, onPress }) => (
    <TouchableOpacity style={styles.quickReply} onPress={onPress}>
      <Text style={styles.quickReplyText}>{text}</Text>
    </TouchableOpacity>
  );

  const quickReplies = [
    'I have a fever',
    'Headache remedy',
    'Diet tips',
    'Exercise advice',
  ];

  return (
    <KeyboardAvoidingView 
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.botHeaderAvatar}>
            <Icon name="shield-checkmark" size={20} color="#fff" />
          </View>
          <View>
            <Text style={styles.headerTitle}>IntelliMed Assistant</Text>
            <Text style={styles.headerStatus}>Online</Text>
          </View>
        </View>
        <View style={styles.headerLogoContainer}>
          <Image 
            source={require('../assets/logo.png')} 
            style={styles.headerLogo} 
            resizeMode="contain" 
          />
        </View>
        <TouchableOpacity 
          style={styles.infoButton}
          onPress={() => Alert.alert(
            'IntelliMed Assistant',
            'I can help with questions about your medical reports, nutrition/diet, exercise, and general health guidelines. Powered by IntelliMed Longitudinal Health Intelligence.'
          )}
        >
          <Icon name="information-circle-outline" size={24} color="#123C58" />
        </TouchableOpacity>
      </View>

      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={item => item.id}
        style={styles.messagesList}
        onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
        onLayout={() => flatListRef.current?.scrollToEnd()}
      />

      {/* Typing indicator */}
      {isTyping && (
        <View style={styles.typingContainer}>
          <View style={styles.botAvatar}>
            <Icon name="shield-checkmark" size={16} color="#fff" />
          </View>
          <View style={styles.typingBubble}>
            <ActivityIndicator size="small" color="#666" />
            <Text style={styles.typingText}>IntelliMed is typing...</Text>
          </View>
        </View>
      )}

      {/* Quick Replies */}
      {messages.length <= 1 && (
        <View style={styles.quickRepliesContainer}>
          <Text style={styles.quickRepliesTitle}>Quick questions:</Text>
          <View style={styles.quickRepliesRow}>
            {quickReplies.map((reply, index) => (
              <QuickReply
                key={index}
                text={reply}
                onPress={() => setInputText(reply)}
              />
            ))}
          </View>
        </View>
      )}

      {/* Input */}
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Type your health question..."
          multiline
          maxLength={500}
        />
        <TouchableOpacity 
          style={[styles.sendButton, { opacity: inputText.trim() ? 1 : 0.5 }]}
          onPress={sendMessage}
          disabled={!inputText.trim()}
        >
          <Icon name="send" size={20} color="#fff" />
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
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
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    paddingTop: 45,
  },
  headerLogoContainer: {
    width: 100,
    height: 30,
  },
  headerLogo: {
    width: '100%',
    height: '100%',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  botHeaderAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#158C86',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  headerStatus: {
    fontSize: 14,
    color: '#4CAF50',
  },
  infoButton: {
    padding: 8,
  },
  messagesList: {
    flex: 1,
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  messageContainer: {
    flexDirection: 'row',
    marginVertical: 4,
  },
  botMessage: {
    alignSelf: 'flex-start',
  },
  userMessage: {
    alignSelf: 'flex-end',
  },
  botAvatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#158C86',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
    marginTop: 8,
  },
  messageBubble: {
    maxWidth: '70%',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 18,
    marginBottom: 4,
  },
  botBubble: {
    backgroundColor: '#fff',
    borderBottomLeftRadius: 4,
  },
  userBubble: {
    backgroundColor: '#123C58',
    borderBottomRightRadius: 4,
    marginLeft: 40,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 20,
  },
  botText: {
    color: '#333',
  },
  userText: {
    color: '#fff',
  },
  timestamp: {
    fontSize: 12,
    marginTop: 4,
  },
  botTimestamp: {
    color: '#999',
  },
  userTimestamp: {
    color: 'rgba(255, 255, 255, 0.8)',
  },
  typingContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  typingBubble: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 18,
    borderBottomLeftRadius: 4,
  },
  typingText: {
    marginLeft: 8,
    color: '#666',
    fontStyle: 'italic',
  },
  quickRepliesContainer: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  quickRepliesTitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  quickRepliesRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  quickReply: {
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    marginBottom: 8,
  },
  quickReplyText: {
    fontSize: 14,
    color: '#333',
  },
  inputContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#fff',
    alignItems: 'flex-end',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    maxHeight: 100,
    marginRight: 8,
    fontSize: 16,
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#123C58',
    justifyContent: 'center',
    alignItems: 'center',
  },
});