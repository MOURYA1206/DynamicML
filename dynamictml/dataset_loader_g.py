import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Binarizer


class Preprocessing_Scikit_Img_Classif_Supervised_MN:

    
    @staticmethod
    def GaussianNB_MN(
        dataset_path = None,
        img_paths = None,
        img_size = (64,64),
        pca = None,
        label_encoder = None,
        mode = "train",
        pca_n_components = 30,
        pca_random_state = 42
    ):
        X , y = [] , []
        def extract(img):
            img = cv2.resize(img,img_size)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = img.astype("float32")/255.0

            return hog(
                img,
                orientations = 8,
                pixels_per_cell = (8,8),
                cells_per_block= (2,2),
                block_norm = "L2-Hys",
                transform_sqrt = False,
                feature_vector = True
            )
        if mode == "train":
            for label in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path,label)
                if not os.path.isdir(class_path):
                    continue
                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path,img_name))
                    if img is None:
                        continue
                    X.append(extract(img))
                    y.append(label)
        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                X.append(extract(img))
        X = np.array(X)
        if pca is None and mode == "train":
            pca = PCA(n_components=pca_n_components,random_state=pca_random_state) 
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)
        
        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                
                return X,y,pca,label_encoder
        return X

    @staticmethod
    def Ridge_Classifier_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(64, 64),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return hog(
                img,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)
        # Scaler
        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        
        if pca is None and mode == "train":
            pca = PCA(n_components=50, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, label_encoder, scaler

        return X

    @staticmethod
    def LinearSVC_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)
        # Scaler
        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        # PCA for LinearSVC(optional)
        if pca is None and mode == "train":
            pca = PCA(n_components=0.95, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, label_encoder , scaler

        return X

    @staticmethod
    def RandomForest_Classifier_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(64, 64),
        label_encoder=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return hog(
                img,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
                transform_sqrt=True
            )

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, label_encoder

        return X
        
    @staticmethod
    def svc_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(64, 64),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray.astype("float32") / 255.0

            return hog(
                gray,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in os.listdir(dataset_path):
                for img_name in os.listdir(os.path.join(dataset_path, label)):
                    img = cv2.imread(os.path.join(dataset_path, label, img_name))
                    if img is not None:
                        X.append(extract(img))
                        y.append(label)
        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is not None:
                    X.append(extract(img))

        X = np.array(X)

        if pca is None and mode == "train":
            pca = PCA(n_components=0.90, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)

        if mode == "train":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            return X, y, (pca, scaler), label_encoder

        return X

    




    @staticmethod
    def knn_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(64, 64),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray.astype("float32") / 255.0

            return hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in os.listdir(dataset_path):
                for img_name in os.listdir(os.path.join(dataset_path, label)):
                    img = cv2.imread(os.path.join(dataset_path, label, img_name))
                    if img is not None:
                        X.append(extract(img))
                        y.append(label)
        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is not None:
                    X.append(extract(img))

        X = np.array(X)

        if pca is None and mode == "train":
            pca = PCA(n_components=50, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)

        if mode == "train":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            return X, y, (pca, scaler), label_encoder

        return X





    # =========================
    # Gaussian Process
    # =========================
    @staticmethod
    def gaussianprocess_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(64, 64),
        pca=None,
        label_encoder=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return hog(
                img,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # Strong PCA for GP
        if pca is None and mode == "train":
            pca = PCA(n_components=30, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, label_encoder

        return X
    
        
    @staticmethod
    def bagging_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(64, 64),
        pca=None,
        label_encoder=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)

            # ----- COLOR FEATURES -----
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color_features = []

            for i in range(3):  # R, G, B
                channel = img_rgb[:, :, i]
                color_features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.max(channel) - np.min(channel)
                ])

            # ----- HOG FEATURES -----
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = gray.astype("float32") / 255.0

            hog_features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

            return np.concatenate([hog_features, color_features])

        # -------------------------
        # DATA LOADING
        # -------------------------
        if mode == "train":
            for label in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)
        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # -------------------------
        # PCA (Light – keep variance)
        # -------------------------
        if pca is None and mode == "train":
            pca = PCA(n_components=0.97, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        # -------------------------
        # LABEL ENCODING
        # -------------------------
        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, label_encoder

        return X



    @staticmethod
    def gradientboosting_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(64, 64),
        pca=None,
        label_encoder=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)
        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # Light PCA for trees
        if pca is None and mode == "train":
            pca = PCA(n_components=0.95, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, label_encoder

        return X


    # =========================
    # AdaBoost
    # =========================
    @staticmethod
    def Adaboost_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(64, 64),
        pca=None,
        label_encoder=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            lbp = local_binary_pattern(
                img,
                P=8,
                R=1,
                method="uniform"
            )

            hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, 11),
                density=True
            )
            return hist

        if mode == "train":
            for label in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)
        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # PCA to stabilize AdaBoost
        if pca is None and mode == "train":
            pca = PCA(n_components=0.98, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, label_encoder

        return X
    

    @staticmethod
    def DecisionTreeClassifier_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return hog(
                img,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)
        # Scaler
        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        # Light PCA 
        if pca is None and mode == "train":
            pca = PCA(n_components=60, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, label_encoder, scaler

        return X
    

    @staticmethod
    def ComplementNB_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return img.flatten()

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # =============================
        # MinMaxScaler (MANDATORY)
        # =============================
        if scaler is None and mode == "train":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, scaler, label_encoder

        return X
    
    @staticmethod
    def MLPClassifier_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0
            return img.flatten()

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # =============================
        # StandardScaler (MANDATORY for MLP)
        # =============================
        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        # =============================
        # Optional PCA (ONLY if passed)
        # =============================
        if pca is not None:
            if mode == "train":
                X = pca.fit_transform(X)
            else:
                X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, scaler, label_encoder

        return X
        
    @staticmethod
    def RadiusNeighborsClassifier_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return hog(
                img,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)
        # Scaler
        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        # Light PCA 
        if pca is None and mode == "train":
            pca = PCA(n_components=40, random_state=42)
            X = pca.fit_transform(X)
        elif pca is not None:
            X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, label_encoder, scaler

        return X
    
    @staticmethod
    def ExtraTreeClassifier_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return hog(
                img,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, label_encoder, scaler

        return X
    
    @staticmethod
    def HistGradientBoostingClassifier_MN(
        dataset_path=None,
        img_size=(128, 128),
        img_paths=None,
        mode = "train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0

            return hog(
                img,
                orientations=9,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                block_norm="L2-Hys"
            )
        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):

                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):

                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

            return np.array(X), np.array(y), None, None


        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))   

            return np.array(X), None     
    
    @staticmethod
    def VotingClassifier_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0
            return img.flatten()

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # =============================
        # Optional Scaling
        # =============================
        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        # =============================
        # Optional PCA
        # =============================
        if pca is not None:
            if mode == "train":
                X = pca.fit_transform(X)
            else:
                X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, scaler, label_encoder

        return X
    
    @staticmethod
    def NearestCentroid_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0
            return img.flatten()

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # =============================
        # Scaling (MANDATORY for distance models)
        # =============================
        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        # =============================
        # Optional PCA
        # =============================
        if pca is not None:
            if mode == "train":
                X = pca.fit_transform(X)
            else:
                X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, scaler, label_encoder

        return X
    
    @staticmethod
    def MultinomialNB_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0
            return img.flatten()

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # ============================
        # MinMaxScaler (MANDATORY)
        # ============================
        if scaler is None and mode == "train":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, scaler, label_encoder

        return X
    
    @staticmethod
    def CategoricalNB_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        label_encoder=None,
        scaler=None,
        pca=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0
            
            return hog(
                img,
                orientations=8,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        if pca is None and mode == "train":
            pca = PCA(n_components=100, random_state=42)
            X = pca.fit_transform(X)

            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        elif pca is not None:
            X = pca.transform(X)
            X = scaler.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, pca, scaler, label_encoder

        return X
    
    @staticmethod
    def BernoulliNB_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        label_encoder=None,
        binarizer=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0
            
            return hog(
                img,
                orientations=8,
                pixels_per_cell=(8,8),
                cells_per_block=(2,2),
                block_norm="L2-Hys"
            )

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        if binarizer is None and mode == "train":
            binarizer = Binarizer(threshold=0.0)
            X = binarizer.fit_transform(X)
        elif binarizer is not None:
            X = binarizer.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, binarizer, label_encoder

        return X
    
    @staticmethod
    def SGDClassifier_MN(
        dataset_path=None,
        img_paths=None,
        img_size=(128, 128),
        pca=None,
        label_encoder=None,
        scaler=None,
        mode="train"
    ):
        X, y = [], []

        def extract(img):
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0
            return img.flatten()

        if mode == "train":
            for label in sorted(os.listdir(dataset_path)):
                class_path = os.path.join(dataset_path, label)
                if not os.path.isdir(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    img = cv2.imread(os.path.join(class_path, img_name))
                    if img is None:
                        continue

                    X.append(extract(img))
                    y.append(label)

        else:
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                X.append(extract(img))

        X = np.array(X)

        # =============================
        # Scaling (MANDATORY for SGD)
        # =============================
        if scaler is None and mode == "train":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler is not None:
            X = scaler.transform(X)

        # =============================
        # Optional PCA
        # =============================
        if pca is not None:
            if mode == "train":
                X = pca.fit_transform(X)
            else:
                X = pca.transform(X)

        if mode == "train":
            if label_encoder is None:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            else:
                y = label_encoder.transform(y)

            return X, y, scaler, label_encoder

        return X