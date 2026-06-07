# DynamicML 🚀  
**Dynamic, model‑aware classical machine learning for image classification**

**CREATED BY SUNKARA SAI GANESH , KARROTHU MOURYA , KUDIRELLA SANMUKA SAI, DONTALA. KIRAN KUMAR.**

Train powerful machine learning models for image classification with built-in preprocessing, feature extraction, scaling, dimensionality reduction, evaluation, and model serialization — all through a clean Python API.
DynamicML is an intelligent image-based ML framework focused on classical machine learning pipelines for computer vision tasks. It automates the difficult parts of image preprocessing using techniques like HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns), PCA, feature scaling, label encoding, and model persistence, allowing developers to focus on experimentation instead of boilerplate code.
The framework currently supports a wide collection of supervised image classification algorithms including SVMs, Random Forests, Boosting models, Naive Bayes, Neural Networks, Ensemble models, and more. 
________________________________________
✨ Features
Category	Capabilities
Image Processing	Automatic image resizing, grayscale conversion, normalization
Feature Extraction	HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns), flattened pixel features
Feature Engineering	PCA dimensionality reduction, StandardScaler, MinMaxScaler
Machine Learning	SVM, Random Forest, RidgeClassifier, KNN, Gradient Boosting, AdaBoost, MLP, GaussianNB, and many more
Evaluation	Accuracy score, classification report, confusion matrix
Model Persistence	Save trained models using Joblib
Dataset Handling	Folder-based image dataset loading
Prediction Pipeline	Reusable preprocessing objects for inference
Classical CV Pipeline	Optimized preprocessing strategies tailored per model
________________________________________
🚀 Why DynamicML?
Traditional image ML workflows usually require manually writing:
•	image preprocessing pipelines 
•	feature extraction logic 
•	PCA transformation 
•	scaling pipelines 
•	label encoding 
•	model saving/loading 
•	evaluation code 
DynamicML combines all of these into a unified framework.
DynamicML	Traditional Workflow
Automatic preprocessing	Manual OpenCV pipelines
Built-in HOG/LBP extraction	Custom feature engineering
Automatic scaling & PCA	Separate preprocessing scripts
One-function model training	Multiple training files
Saved preprocessing bundles	Manual artifact management
Ready for inference	Rebuild preprocessing pipeline manually
________________________________________
🧠 Supported Models
DynamicML currently includes support for:
Classification Models
•	GaussianNB 
•	ComplementNB 
•	MultinomialNB 
•	RidgeClassifier 
•	LinearSVC 
•	SVC 
•	NuSVC 
•	RandomForestClassifier 
•	DecisionTreeClassifier 
•	ExtraTreesClassifier 
•	GradientBoostingClassifier 
•	HistGradientBoostingClassifier 
•	AdaBoostClassifier 
•	BaggingClassifier 
•	VotingClassifier 
•	KNeighborsClassifier 
•	RadiusNeighborsClassifier 
•	GaussianProcessClassifier 
•	MLPClassifier 
•	NearestCentroid 
Implemented inside the training pipeline system. 
________________________________________
🖼️ Image Processing Pipeline
DynamicML uses intelligent preprocessing strategies depending on the model type.
HOG Feature Extraction
DynamicML heavily utilizes:
•	HOG (Histogram of Oriented Gradients) 
•	PCA dimensionality reduction 
•	Standard scaling 
•	grayscale normalization 
Example HOG extraction pipeline: 
LBP Texture Features
For texture-sensitive models like AdaBoost, DynamicML uses:
•	Local Binary Patterns (LBP) 
•	histogram-based texture encoding 
Implemented preprocessing: 
________________________________________
📦 Installation
pip install dynamicml
Development Installation
git clone https://github.com/yourusername/dynamicml.git

cd dynamicml

pip install -e .
________________________________________
📂 Dataset Structure
DynamicML expects datasets in folder-based format:
dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
Each folder name becomes the class label automatically using LabelEncoder. 
________________________________________
⚡ Quick Start
Train a Random Forest Classifier
from dynamicml import Scikit_Img_Classif_Supervised

model = Scikit_Img_Classif_Supervised.RandomForest_Classifier(
    dataset_path="dataset/"
)
________________________________________
Train an SVC Model
from dynamicml import Scikit_Img_Classif_Supervised

model = Scikit_Img_Classif_Supervised.SVC(
    dataset_path="dataset/",
    C_val=5.0,
    kernel_val="rbf"
)
________________________________________
Train a Neural Network
from dynamicml import Scikit_Img_Classif_Supervised

model = Scikit_Img_Classif_Supervised.MLP_Classifier(
    dataset_path="dataset/",
    hidden_layer_sizes_val=(256, 128),
    max_iter_val=500
)
________________________________________
📊 Example Output
Loading and preprocessing data...
Training Random Forest Classifier...
Fitting the Model....
Model Training Completed!

Accuracy: 96.4%

Classification Report:

              precision    recall  f1-score   support

       cats       0.97      0.95      0.96
       dogs       0.96      0.98      0.97

Confusion Matrix:

[[95  5]
 [ 2 98]]

Model saved at:
RandomForestClassifier_MN.joblib
________________________________________
🧩 Architecture
Dataset
    ↓
Preprocessing 
    ↓ 
Classifier (Single Model)
    ↓ 
Evaluation 
    ↓ 
Model Saving
________________________________________
🔬 Core Technologies Used
DynamicML is powered by:
•	OpenCV 
•	scikit-learn 
•	NumPy 
•	Joblib 
•	scikit-image 
________________________________________
🧠 Intelligent Preprocessing
Different models use different optimized pipelines internally.
Examples:
Model	Preprocessing Strategy
SVC	HOG + PCA + StandardScaler
RandomForest	HOG features
AdaBoost	LBP texture features
MLP	Flattened pixels + StandardScaler
KNN	HOG + PCA + Scaling
GaussianNB	HOG + PCA
Implemented dynamically inside preprocessing pipelines. 
________________________________________
💾 Model Saving
DynamicML automatically saves:
•	trained model 
•	PCA object 
•	scaler 
•	label encoder 
using Joblib bundles.
Example save pipeline: 
________________________________________
📈 Evaluation Metrics
DynamicML automatically provides:
•	Accuracy Score 
•	Classification Report 
•	Confusion Matrix 
Integrated evaluation helper: 
________________________________________
🎯 Project Vision
DynamicML aims to become a complete classical computer vision machine learning framework that simplifies image classification workflows while still giving developers deep control over preprocessing and model configuration.
The goal is to make traditional ML for computer vision:
•	faster 
•	cleaner 
•	reusable 
•	production-friendly 
•	beginner accessible 
without sacrificing flexibility.
________________________________________
🛣️ Roadmap
Planned Features
•	Prediction API 
•	Batch inference system 
•	Deep learning integration 
•	Automatic model selection 
•	Hyperparameter tuning 
•	Experiment tracking 
•	Streamlit dashboard 
•	CLI interface 
•	ONNX export 
•	FastAPI deployment 
•	GPU acceleration 
•	Explainable AI visualizations 
________________________________________
🤝 Contributing
Contributions are welcome!
You can contribute by:
•	adding new ML models 
•	improving preprocessing pipelines 
•	optimizing feature extraction 
•	improving documentation 
•	adding deployment integrations 
________________________________________
📜 License
DynamicML is released under the MIT License.
