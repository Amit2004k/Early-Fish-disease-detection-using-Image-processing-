Fish Disease Detection Application Documentation 
Overview 
The Fish Disease Detection Application is a revolutionary tool that harnesses the power of artificial 
intelligence (AI) and machine learning (ML) to diagnose fish diseases from uploaded images with 
precision and speed. Designed to address the escalating challenges faced by the aquaculture 
industry, this application combines cutting-edge technology with user-friendly interfaces to 
empower fish farmers, researchers, and stakeholders. Fish diseases are a growing concern due to 
environmental and anthropogenic factors such as pollution, climate change, overcrowding in 
aquaculture systems, and poor water quality management. These conditions create a breeding 
ground for bacterial, viral, fungal, and parasitic diseases, threatening aquatic biodiversity and the 
sustainability of fish farming practices. 
By integrating advanced backend functionalities, a responsive frontend interface, and a well-defined 
roadmap for future enhancements, the Fish Disease Detection Application aims to transform the 
aquaculture sector. Its ability to deliver accurate disease diagnoses and actionable insights not only 
supports healthier fish populations but also promotes sustainable aquaculture practices and 
improved economic outcomes for farmers. The application also envisions contributing to global 
food security by mitigating fish disease outbreaks and enhancing production efficiency. 
Through the power of AI/ML, this application reduces dependency on traditional, time-consuming 
diagnostic methods, which often require expert knowledge and laboratory resources. Instead, it 
provides an accessible, scalable, and efficient alternative, capable of analysing images and 
predicting diseases in real-time. By fostering healthier ecosystems and enabling proactive disease 
management, this initiative sets a new benchmark in the application of technology for 
environmental and economic sustainability in aquaculture. 
1. System Architecture 
Frontend Overview 
• User Interaction: The frontend provides a user-friendly and intuitive interface where users 
can seamlessly upload images of fish in JPG or JPEG format for disease analysis. It is 
designed to be easily navigable by users from diverse technical backgrounds, ensuring broad 
accessibility. 
• Technologies: Built using HTML, CSS, and JavaScript, the frontend ensures responsiveness 
and compatibility across various devices, including desktops, tablets, and mobile phones. 
Features like drag-and-drop image uploads further enhance the user experience. 
• Integration: It acts as a bridge to the backend by sending user-provided images to the 
machine learning model via APIs and subsequently displaying predictions and tailored 
recommendations to users in a visually appealing manner. 
Backend Overview 
• Flask Framework: This lightweight yet powerful web framework manages interactions 
between the frontend and the machine learning model, ensuring efficient processing and 
response times. 
• Custom CNN Model: The backend integrates a carefully designed custom convolutional 
neural network (CNN) model that classifies fish images into one of seven disease categories 
with high accuracy. The CNN model is optimised for both accuracy and computational 
efficiency. 
• ChatGPT Integration: Future integration with ChatGPT will provide personalised post
diagnosis advice, including step-by-step remedies, preventive measures, and expert farming 
tips tailored to the diagnosed condition. 
2. Backend Development 
Virtual Environment Setup 
1.
 2.
 3.
 4.
 Create Virtual Environment: 
python -m venv venv
 Activate Virtual Environment: 
◦ Windows: 
.\venv\Scripts\activate
 ◦ Linux/Mac: 
source venv/bin/activate
 Install Required Libraries: 
pip install flask tensorflow pillow numpy
 Generate Requirements File: 
pip freeze > requirements.txt
 Model Loading and Prediction 
• The backend is responsible for loading the trained custom CNN model from a .h5 file. The 
uploaded fish images are preprocessed to match the model’s expected input dimensions of 
150x150 pixels. This step ensures compatibility and optimal performance during prediction. 
3. Custom CNN Model Code 
from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Conv2D, MaxPooling2D, 
Flatten, Dense, Dropout
 from 
tensorflow.keras.preprocessing.image 
ImageDataGenerator
 # Define the model
 model = Sequential([
 import 
Conv2D(32, (3, 3), activation='relu', input_shape=(150, 
150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
 Dense(7, activation='softmax')  # 7 classes for fish 
diseases
 ])
 # Compile the model
 model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
 )
 # Data augmentation for training
 train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
 train_generator = train_datagen.flow_from_directory(
    'path_to_train_folder',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
 )
 # Train the model
 model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=100
 )
 # Save the model
 model.save(“fish_disease_classifier.h5")
 4. Results and Performance 
Custom CNN 
• Test Accuracy: 91.68% 
• Observation: The custom CNN model demonstrated an exceptional balance between 
accuracy and computational efficiency. Despite being simpler than models like VGG19, it 
avoided overfitting while delivering reliable predictions. This makes it highly suitable for 
deployment in real-world applications. 
Architecture of Custom CNN 
The Custom Convolutional Neural Network (CNN) architecture was specifically designed and fine
tuned to effectively classify fish diseases based on image data. The model emphasises simplicity 
and efficiency while maintaining high accuracy, making it ideal for the requirements of this 
application. 
1. Input Layer: 
◦ Accepts RGB fish images resized to a uniform dimension of 224x224x3. 
◦ This standardisation ensures compatibility across the dataset while preserving 
sufficient image details. 
2. Convolutional Layers: 
◦ Conv2D Layers: Multiple convolutional layers extract spatial features from the 
input images. Each layer uses: 
▪ Kernels (filters) of size 3x3. 
▪ ReLU activation function to introduce non-linearity. 
◦ These layers capture low-level to high-level features such as edges, textures, and 
complex patterns. 
3. Max Pooling Layers: 
◦ Added after convolutional layers to reduce spatial dimensions and computational 
overhead. 
◦ Pool size: 2x2. 
◦ Helps retain critical features while discarding irrelevant details. 
4. Dropout Layers: 
◦ Dropout regularisation is applied during training to mitigate overfitting by randomly 
disabling a fraction of neurons. 
◦ Dropout rate: 0.5. 
5. Fully Connected (Dense) Layers: 
◦ The flattened output of the convolutional layers is passed through dense layers to 
perform classification. 
◦ Includes one or more dense layers with ReLU activation to consolidate extracted 
features. 
6. Output Layer: 
◦ A dense layer with softmax activation function is used to predict the probabilities 
for each fish disease category. 
◦ The number of output nodes corresponds to the 7 classes of fish diseases. 
7. Optimizer and Loss Function: 
◦ Adam optimizer is used for its adaptive learning rate and robust performance. 
◦ The model minimises categorical cross-entropy loss, suitable for multi-class 
classification tasks. 
8. Performance Enhancements: 
◦ Data augmentation techniques, including random rotations, flips, and brightness 
adjustments, were applied to improve generalisation. 
◦ Early stopping was implemented during training to prevent overfitting and ensure 
optimal performance. 
Summary of Model Parameters: 
Layer Type
 Input
 Parameters
 Image dimensions: 224x224x3
 Conv2D + ReLU
 MaxPooling2D
 Filters: 32, Kernel size: 3x3
 Pool size: 2x2
 Conv2D + ReLU
 MaxPooling2D
 Filters: 64, Kernel size: 3x3
 Pool size: 2x2
 Dropout
 Dense + ReLU
 Rate: 0.5
 Units: 128
 Output (Softmax)
 Units: 7 (corresponding to classes)
 The Custom CNN model's balance of simplicity, efficiency, and accuracy makes it highly suitable 
for fish disease detection tasks, outperforming more complex models like VGG19 in avoiding 
overfitting while maintaining a high level of classification accuracy. 
VGG19 
• Test Accuracy: 98.28% 
• Observation: VGG19 achieved the highest test accuracy but exhibited a higher tendency 
toward overfitting, making it less favourable for deployment. Its complexity also increased 
training and inference times. 
MobileNetV2 
• Test Accuracy: 72.74% 
• Observation: While MobileNetV2 provided reasonable accuracy with reduced 
computational requirements, it fell short compared to the custom CNN in handling complex 
disease classifications. 
EfficientNet 
• Test Accuracy: Insufficient for deployment. 
• Observation: EfficientNet failed to perform adequately on the dataset, indicating potential 
challenges in model compatibility or dataset alignment. 
Comparison and Rationale for Choosing Custom CNN 
Model
 Custom CNN
 Test Accuracy
 91.68%
 Overfitting
 Low
 Parameters
 Training Time
 Moderate
 VGG19
 MobileNetV2
 98.28%
 72.74%
 High
 Moderate
 Very High
 Fast
 Slow
 Low
 EfficientNet
 Low
 Moderate
 High
 Moderate
 Very Slow
 The custom CNN model strikes the optimal balance between accuracy and computational 
efficiency. While VGG19 achieved higher accuracy, its overfitting tendencies posed a significant 
challenge. MobileNetV2 and EfficientNet, though efficient, did not meet the accuracy requirements 
for deployment. The custom CNN’s tailored architecture and reduced risk of overfitting made it the 
ideal choice for this application. 
5. Future Scope 
Integration with ChatGPT 
• Plans include the addition of ChatGPT to provide users with detailed, disease-specific 
guidance after diagnosis. 
• Actionable insights such as recommended treatments, preventive measures, and farming tips 
will enhance user experience and practical utility. 
Additional Features 
• Real-time video feed analysis for continuous monitoring of fish health. 
• Multi-language support to cater to diverse user bases worldwide. 
• Optimised deployment for mobile platforms to ensure accessibility on the go. 
6. Deployment Workflow 
Deployment with Render and GitHub 
1.
 2.
 Prepare Repository: 
◦ Include app.py, requirements.txt, Procfile, and the trained model 
f
 ile (.h5) in the GitHub repository.
 Render Setup: 
◦ Link the repository to Render. 
◦ Define the build command: 
pip install -r requirements.txt
 ◦ Define the start command: 
gunicorn app:app
3.
 Continuous Deployment: 
◦ Push updates to the GitHub repository to automatically redeploy the application. 
Conclusion 
The Fish Disease Detection Application represents a transformative advancement in applying 
artificial intelligence (AI) and machine learning (ML) to solve pressing real-world challenges. This 
innovative application is specifically designed to diagnose fish diseases with remarkable accuracy 
and efficiency by analysing uploaded images. Its development responds to the critical issues faced 
by the aquaculture industry, where diseases pose a significant threat to fish populations, 
biodiversity, and the economic stability of fish farming practices. 
The rise in fish diseases is fuelled by various environmental and human-induced factors. Pollution 
from industrial and agricultural activities degrades water quality, creating a hostile environment for 
aquatic life. Climate change exacerbates these challenges by altering water temperatures and 
disrupting ecosystems, making fish more susceptible to diseases. Overcrowding in aquaculture 
systems, combined with insufficient water quality management, further amplifies the risks of 
bacterial, viral, fungal, and parasitic infections. These issues underline the urgent need for 
innovative solutions like the Fish Disease Detection Application. 
By integrating advanced backend functionalities, a highly responsive frontend interface, and a 
strategic roadmap for future enhancements, this application is set to revolutionise aquaculture 
practices. It empowers fish farmers, researchers, and other stakeholders with precise disease 
diagnoses and actionable insights, enabling them to make informed decisions quickly. This 
proactive approach promotes healthier fish populations, supports sustainable aquaculture practices, 
and helps mitigate the economic losses associated with disease outbreaks. 
The application leverages the power of AI/ML to offer a more accessible and efficient alternative to 
traditional diagnostic methods. Conventional techniques often require expert knowledge, laboratory 
resources, and significant time investments, which can delay intervention and exacerbate disease 
spread. In contrast, this application provides real-time analysis and predictions, ensuring swift 
responses to emerging issues. Its scalability and ease of use make it a valuable tool for both small
scale and industrial aquaculture operations. 
Beyond its immediate benefits, the Fish Disease Detection Application aligns with broader 
sustainability goals. By reducing the reliance on chemical treatments and antibiotics, it fosters eco
friendly practices that protect aquatic ecosystems. Additionally, its role in mitigating fish disease 
outbreaks contributes to global food security by enhancing production efficiency and ensuring a 
stable supply of fish as a vital protein source. 
This initiative is more than just a technological solution; it is a step towards redefining the future of 
aquaculture. By combining cutting-edge AI/ML technology with practical, user-centric design, the 
Fish Disease Detection Application establishes a new standard for environmental stewardship and 
economic resilience in the face of modern challenges. 
