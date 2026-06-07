from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os
from dynamictml.dataset_loader_g import Preprocessing_Scikit_Img_Classif_Supervised_MN


class Scikit_Img_Classif_Supervised:

    # =========================
    # Common helpers
    # =========================
    @staticmethod
    def _save_bundle(bundle, prefix, save_dir):
        if save_dir is None:
            save_dir = os.getcwd()

        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(save_dir, f"{prefix}.joblib")

        joblib.dump(bundle, path)
        print(f"Model saved at: {path}")
        return path

    @staticmethod
    def _evaluate(y_test, y_pred):
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    @staticmethod
    def GaussianNB(dataset_path , test_size_val = 0.2,random_state=42,priors_val = None , var_smoothing_val = 1e-9 , save_dir = None, save_name="GaussianNB_MN",pca_n_components = 30, pca_random_state = 40):
        from sklearn.naive_bayes import GaussianNB
        try:
            print("Loading and Preprocessing data....")
            X, y , pca , label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.GaussianNB_MN(dataset_path=dataset_path,pca_n_components=pca_n_components,pca_random_state=pca_random_state)
            X_train ,X_test , y_train , y_test = train_test_split(X, y , test_size=test_size_val,random_state=random_state)
            print("Training GaussianNB...")
            gnb = GaussianNB(priors=priors_val,
                             var_smoothing= var_smoothing_val,
                             )
            print("Fitting the model....")
            gnb.fit(X_train,y_train)
            print("Model training completed....")
            y_pred = gnb.predict(X_test)
            Scikit_Img_Classif_Supervised._evaluate(y_test,y_pred)
            bundle = {
                "model": gnb,
                "pca": pca,
                "label_encoder": label_encoder
            }
            Scikit_Img_Classif_Supervised._save_bundle(bundle=bundle,prefix=save_name,save_dir=save_dir)
            print(" GaussianNB training, evaluation, and model saving completed successfully.")
            return bundle
        except Exception as e:
            print(f"Unknown Exception happened: {e}")

    
    @staticmethod
    def RidgeClassifier(dataset_path, test_size_val=0.2, split_random_state_val=42, shuffle_val=True,save_name = "RidgeClassifier_MN",
                                                alpha_val=1.0, fit_intercept_val=True, copy_X_val=True, max_iter_val=None,
                                                tol_val=0.0001, class_weight_val=None, solver_val='auto',
                                                positive_val=False, random_state_val=None, save_dir=None):
        from sklearn.linear_model import RidgeClassifier
        try:
            print("Loading and pre-processing data....")
            X , y , pca , label_encoder , Scaler =Preprocessing_Scikit_Img_Classif_Supervised_MN.Ridge_Classifier_MN(dataset_path)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_val, random_state=split_random_state_val,
                stratify=y, shuffle=shuffle_val
            )
            print("Training RidgeClassifier...")
            rc = RidgeClassifier(
                alpha=alpha_val,
                fit_intercept=fit_intercept_val,
                copy_X=copy_X_val,
                max_iter=max_iter_val,
                tol=tol_val,
                class_weight=class_weight_val,
                solver=solver_val,
                positive=positive_val,
                random_state=random_state_val
            )
            print("Fitting the Model....")
            rc.fit(X_train, y_train)
            print("Model Training Completed!")
            y_pred = rc.predict(X_test)

            print("Accuracy score:",accuracy_score(y_test,y_pred))
            print("\nClassification Report:\n",classification_report(y_test,y_pred))

            print("Saving the Model....")
            bundle = {
                "model": rc,
                "pca": pca,
                "label_encoder": label_encoder,
                "scaler": Scaler
            }
            Scikit_Img_Classif_Supervised._save_bundle(bundle=bundle,prefix=save_name,save_dir=save_dir)
            print(f"Model saved at: {save_dir}")
            print(" RidgeClassifier training, evaluation, and model saving completed successfully.")
            return rc
        except Exception as e:
            print(f"An error occured: {e}")
            return None

    @staticmethod
    def LinearSVC(dataset_path,test_size_val=0.2,split_random_state_val=42,shuffle_val=True,save_name="LinearSVC_MN",
                  penalty_val='l2', loss_val='squared_hinge', dual_val='auto', tol_val=0.0001,
                  C_val=1.0, multi_class_val='ovr', fit_intercept_val=True, intercept_scaling_val=1,
                  class_weight_val=None, verbose_val=0, random_state_val=None, max_iter_val=1000, save_dir=None
            ):
        
        from sklearn.svm import LinearSVC

        try:
            print("Loading and pre-processing data....")
            X , y , pca, label_encoder , scaler =Preprocessing_Scikit_Img_Classif_Supervised_MN.LinearSVC_MN(dataset_path)
            X_train, X_test, y_train, y_test = train_test_split(
                X , y , test_size=test_size_val, random_state=split_random_state_val,
                stratify=y , shuffle=shuffle_val
            )
            print("Training LinearSVC...")
            model= LinearSVC(
                penalty=penalty_val,
                loss=loss_val,
                dual=dual_val,
                tol=tol_val,
                C=C_val,
                multi_class=multi_class_val,
                fit_intercept=fit_intercept_val,
                intercept_scaling=intercept_scaling_val,
                class_weight=class_weight_val,
                verbose=verbose_val,
                random_state=random_state_val,
                max_iter=max_iter_val
            )
            print("Fitting the Model....")
            model.fit(X_train,y_train)
            print("Model Training Completed!")
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test,y_pred)

            bundle = {
                "model": model,
                "pca":pca,
                "label_encoder": label_encoder,
                "scaler":scaler
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle , save_name ,save_dir
            )
            print(" LinearSVC training, evaluation, and model saving completed successfully.")
            return model
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    @staticmethod
    def SVC(dataset_path, test_size_val=0.2,split_random_state_val=42, shuffle_val=True,
                                    C_val=5.0, kernel_val='rbf', degree_val=3, gamma_val='scale', coef0_val=0.0,
                                    shrinking_val=True, probability_val=False, tol_val=1e-3, cache_size_val=200,
                                    class_weight_val='balanced', verbose_val=False, max_iter_val=-1, 
                                    decision_function_shape_val='ovr', break_ties_val=False,random_state_val=None, save_dir=None,save_name="SVC_MN"):
        
        from sklearn.svm import SVC

        try:
            print("Loading and pre-processing data....")
            X , y , pca , label_encoder =Preprocessing_Scikit_Img_Classif_Supervised_MN.svc_MN(dataset_path)
            X_train, X_test, y_train, y_test = train_test_split(
                X , y , test_size=test_size_val, random_state=split_random_state_val,
                stratify=y , shuffle=shuffle_val
            )
            print("Training SVC...")
            svc= SVC(
                C=C_val,
                kernel=kernel_val,
                degree=degree_val,
                gamma=gamma_val,
                coef0=coef0_val,
                shrinking=shrinking_val,
                probability=probability_val,
                tol=tol_val,
                cache_size=cache_size_val,
                class_weight=class_weight_val,
                verbose=verbose_val,
                max_iter=max_iter_val,
                decision_function_shape=decision_function_shape_val,
                break_ties=break_ties_val,
                random_state=random_state_val,
            )
            print("Fitting the Model....")
            svc.fit(X_train,y_train)
            print("Model Training Completed!")
            y_pred = svc.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test,y_pred)

            bundle = {
                "model": svc,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle , save_name,save_dir
            )
            print(" SVC training, evaluation, and model saving completed successfully.")
            return svc
        except Exception as e:
            print(f"An error occurred: {e}")
            return None



    @staticmethod
    def RandomForestClassifier(dataset_path, split_random_state_val=42, test_size_val=0.2, shuffle_val=True,save_name ="RandomForestClassifier_MN",
                                                            n_estimators_val=100, criterion_val='gini', max_depth_val=None,
                                                            min_samples_split_val=2, min_samples_leaf_val=1, min_weight_fraction_leaf_val=0.0,
                                                            max_features_val='sqrt', max_leaf_nodes_val=None, min_impurity_decrease_val=0.0,
                                                            bootstrap_val=True, oob_score_val=False, n_jobs_val=None,random_state_val=None,
                                                            verbose_val=0, warm_start_val=False, class_weight_val=None,ccp_alpha_val=0.0,
                                                            max_samples_val=None, monotonic_cst_val=None,save_dir=None):
        
        from sklearn.ensemble import RandomForestClassifier
        try:
            print("Loading and pre-processing data....")
            X , y , label_encoder= Preprocessing_Scikit_Img_Classif_Supervised_MN.RandomForest_Classifier_MN(dataset_path)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_val, random_state=split_random_state_val,
                stratify=y , shuffle=shuffle_val
            )
            print("Training RandomForestClassifier...")
            model = RandomForestClassifier(
                n_estimators=n_estimators_val,
                criterion=criterion_val,
                max_depth=max_depth_val,
                min_samples_split=min_samples_split_val,
                min_samples_leaf=min_samples_leaf_val,
                min_weight_fraction_leaf=min_weight_fraction_leaf_val,
                max_features=max_features_val,
                max_leaf_nodes=max_leaf_nodes_val,
                min_impurity_decrease=min_impurity_decrease_val,
                bootstrap=bootstrap_val,
                oob_score=oob_score_val,
                n_jobs=n_jobs_val,
                random_state=random_state_val,
                verbose=verbose_val,
                warm_start=warm_start_val,
                class_weight=class_weight_val,
                ccp_alpha=ccp_alpha_val,
                max_samples=max_samples_val,
                monotonic_cst=monotonic_cst_val
                )
            print("Fitting the Model....")
            model.fit(X_train, y_train)
            print("Model Training Completed!")
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test,y_pred)

            bundle = {
                "model": model,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle , save_name ,save_dir
            )
            print(" RandomForestClassifier training, evaluation, and model saving completed successfully.")
            return bundle
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    
    @staticmethod
    def NuSVC(dataset_path, test_size_val = 0.2, save_name="NuSVC_MN", random_state_val = 42, kernel_val= 'rbf',
                                        nu_val = 0.2, degree_val=3,gamma_val ='scale',coef0_val=0.0,shrinking_val=True,
                                        probability_val=False,tol_val=1e-3,cache_size_val=200,class_weight_val=None,
                                        verbose_val=False,max_iter_val=-1,decision_function_shape_val='ovr',break_ties=False,save_dir=None
                                            ):

        from sklearn.svm import NuSVC

        try:
            print("Loading and pre-processing data....")
            X, y , pca , label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.svc_MN(dataset_path)
            print("Samples:", X.shape[0])
            print("Features per image:", X.shape[1])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_val, random_state=random_state_val
            )
            print("Training NuSVC...")
            nusvc = NuSVC(
                kernel=kernel_val,
                nu=nu_val,
                degree=degree_val,
                gamma=gamma_val,
                coef0=coef0_val,
                shrinking=shrinking_val,
                probability=probability_val,
                tol=tol_val,
                cache_size=cache_size_val,
                class_weight=class_weight_val,
                verbose=verbose_val,
                max_iter=max_iter_val,
                decision_function_shape=decision_function_shape_val,
                break_ties=break_ties
            )
            print("Fitting the Model....")
            nusvc.fit(X_train, y_train)
            print("Model Training Completed!")
            y_pred = nusvc.predict(X_test)

            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

            Scikit_Img_Classif_Supervised._evaluate(y_test , y_pred)

            bundle={
                "model": nusvc,
                "pca" : pca,
                "label_encoder":label_encoder,
                
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name , save_dir
            )
            print(" NuSVC training, evaluation, and model saving completed successfully.")
            return bundle
        except Exception as e:
            print(f"An error occured:{e}")
            return None


    @staticmethod
    def BaggingClassifier(dataset_path, test_size_val=0.2, split_random_state_val=42, shuffle_val=True,
                                                    estimator_val=None, n_estimators_val=10, max_samples_value=1.0, max_features_val=1.0,
                                                    bootstrap_val=True, bootstrap_features_val=False, oob_score_val=False, warm_start_val=False,
                                                    n_jobs_val=None, random_state_val=None, verbose_val=0,save_dir=None,save_name = "BaggingClassifier_MN"):
        
        
        from sklearn.ensemble import BaggingClassifier

        try:
            print("Loading and pre-processing data....")
            X , y ,pca,label_encoder= Preprocessing_Scikit_Img_Classif_Supervised_MN.bagging_MN(dataset_path=dataset_path, mode="train")
            X_train, X_test, y_train, y_test =train_test_split(
                X , y, test_size=test_size_val, random_state=split_random_state_val,
                stratify=y, shuffle=shuffle_val
            )
            print("Training BaggingClassifier...")
            bc =  BaggingClassifier(
                estimator=estimator_val,
                n_estimators=n_estimators_val,
                max_samples=max_samples_value,
                max_features=max_features_val,
                bootstrap=bootstrap_val,
                bootstrap_features=bootstrap_features_val,
                oob_score=oob_score_val,
                warm_start=warm_start_val,
                n_jobs=n_jobs_val,
                random_state=random_state_val,
                verbose=verbose_val
            )
            print("Fitting the Model....")
            bc.fit(X_train,y_train)
            print("Model Training Completed!")
            y_pred = bc.predict(X_test)

            print("Accuracy score:",accuracy_score(y_test,y_pred))
            print("Classification report:",classification_report(y_test,y_pred))

            bundle = {
                "model": bc,
                "pca": pca,
                "label_encoder": label_encoder
            }
            print("Saving the Model....")
            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" BaggingClassifier training, evaluation, and model saving completed successfully.")

            return bundle
        except Exception as e:
             print(f"An error occured: {e}")
             return None

    @staticmethod
    def KNeighborsClassifier(dataset_path, random_state_val=42, test_size_val=0.2, n_neighbors_val=3, weights_val ='distance', algorithm_val ='brute', leaf_size_val =30,p_val=2, metric_val='euclidean', metric_params_val =None, n_jobs_val=-1,save_dir=None ,save_name = "KNeighborsClassifier_MN"):
        
        from sklearn.neighbors import KNeighborsClassifier
        
        try:
            print("Loading and pre-processing data....")
            X , y ,pca , label_encoder= Preprocessing_Scikit_Img_Classif_Supervised_MN.knn_MN(dataset_path=dataset_path, mode="train")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_val, random_state=random_state_val,
                stratify=y  
            )

            print("Training KNeighborsClassifier...")
            knn = KNeighborsClassifier(
                n_neighbors=n_neighbors_val,
                weights=weights_val,
                algorithm=algorithm_val,
                leaf_size=leaf_size_val,
                p=p_val,
                metric=metric_val,
                metric_params=metric_params_val,
                n_jobs=n_jobs_val
                )
            print("Fitting the Model....")
            knn.fit(X_train, y_train)
            print("Model Training Completed!")
            y_pred = knn.predict(X_test)

            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("\nClassification Report:\n", classification_report(y_test, y_pred))
            bundle = {
                 "model": knn,
                 "pca": pca,
                 "label_encoder": label_encoder
            }
            print("Saving the Model....")
            Scikit_Img_Classif_Supervised._save_bundle(bundle, save_name, save_dir)
            print(" KNeighborsClassifier training, evaluation, and model saving completed successfully.")
            return bundle
        except Exception as e:
            print(f"An error occured: {e}")
            return None
        

    @staticmethod
    def DecisionTreeClassifier(dataset_path, test_size_val=0.2,split_random_state_val=42, shuffle_val=True,
                               criterion_val='gini', splitter_val='best', max_depth_val=None, min_samples_split_val=2,
                               min_samples_leaf_val=1, min_weight_fraction_leaf_val=0.0, max_features_val=None,
                               random_state_val=None, max_leaf_nodes_val=None, min_impurity_decrease_val=0.0,
                               class_weight_val=None, ccp_alpha_val=0.0, monotonic_cst_val=None, save_dir=None):
        
        from sklearn.tree import DecisionTreeClassifier

        try:
            print("Loading and pre-processing data....")
            X , y , pca , label_encoder , scaler =Preprocessing_Scikit_Img_Classif_Supervised_MN.DecisionTreeClassifier_MN(dataset_path)
            X_train, X_test, y_train, y_test = train_test_split(
                X , y , test_size=test_size_val, random_state=split_random_state_val,
                stratify=y , shuffle=shuffle_val
            )
            print("Training DecisionTreeClassifier...")
            model= DecisionTreeClassifier(
                criterion=criterion_val,
                splitter=splitter_val,
                max_depth=max_depth_val,
                min_samples_split=min_samples_split_val,
                min_samples_leaf=min_samples_leaf_val,
                min_weight_fraction_leaf=min_weight_fraction_leaf_val,
                max_features=max_features_val,
                random_state=random_state_val,
                max_leaf_nodes=max_leaf_nodes_val,
                min_impurity_decrease=min_impurity_decrease_val,
                class_weight=class_weight_val,
                ccp_alpha=ccp_alpha_val,
                monotonic_cst=monotonic_cst_val
                
            )
            print("Fitting the Model....")
            model.fit(X_train,y_train)
            print("Model Training Completed!")
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test,y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder,
                "scaler":scaler
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle , "DecisionTreeClassifier_MN",save_dir
            )
            print(" DecisionTreeClassifier training, evaluation, and model saving completed successfully.")
            return bundle
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    @staticmethod
    def GaussianProcessClassifier(
        dataset_path,
        test_size_val=0.2,
        split_random_state_val=42,
        shuffle_val=True,
        kernel_val=None,
        optimizer_val="fmin_l_bfgs_b",
        n_restarts_optimizer_val=0,
        max_iter_predict_val=100,
        warm_start_val=False,
        copy_X_train_val=True,
        random_state_val=None,
        multi_class_val="one_vs_rest",
        n_jobs_val=None,
        save_dir=None,
        save_name="GaussianProcessClassifier_MN"
    ):
        from sklearn.gaussian_process import GaussianProcessClassifier

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.gaussianprocess_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=split_random_state_val,
                stratify=y,
                shuffle=shuffle_val
            )

            print("Training GaussianProcessClassifier...")
            model = GaussianProcessClassifier(
                kernel=kernel_val,
                optimizer=optimizer_val,
                n_restarts_optimizer=n_restarts_optimizer_val,
                max_iter_predict=max_iter_predict_val,
                warm_start=warm_start_val,
                copy_X_train=copy_X_train_val,
                random_state=random_state_val,
                multi_class=multi_class_val,
                n_jobs=n_jobs_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" GaussianProcessClassifier training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    # =========================
    # Gradient Boosting
    # =========================
    @staticmethod
    def GradientBoostingClassifier(
        dataset_path,
        test_size_val=0.2,
        random_state_val=42,
        n_estimators_val=100,
        learning_rate_val=0.1,
        loss_val="log_loss",
        max_depth_val=3,
        min_samples_split_val=10,
        min_samples_leaf_val=5,
        max_features_val="sqrt",
        sub_sample_val=0.8,
        save_dir=None,
        save_name="GradientBoostingClassifier_MN"
    ):
        from sklearn.ensemble import GradientBoostingClassifier

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.gradientboosting_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=random_state_val,
                stratify=y
            )

            print("Training GradientBoostingClassifier...")
            model = GradientBoostingClassifier(
                n_estimators=n_estimators_val,
                learning_rate=learning_rate_val,
                loss=loss_val,
                max_depth=max_depth_val,
                min_samples_split=min_samples_split_val,
                min_samples_leaf=min_samples_leaf_val,
                max_features=max_features_val,
                subsample=sub_sample_val,
                random_state=random_state_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" GradientBoostingClassifier training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    # =========================
    # AdaBoost
    # =========================
    @staticmethod
    def AdaBoostClassifier(
        dataset_path,
        split_random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        estimator_val=None,
        n_estimators_val=50,
        learning_rate_val=1.0,
        random_state_val=None,
        save_dir=None,
        save_name="AdaBoostClassifier_MN"
    ):
        from sklearn.ensemble import AdaBoostClassifier

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.Adaboost_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=split_random_state_val,
                stratify=y,
                shuffle=shuffle_val
            )

            print("Training AdaBoostClassifier...")
            model = AdaBoostClassifier(
                estimator=estimator_val,
                n_estimators=n_estimators_val,
                learning_rate=learning_rate_val,
                random_state=random_state_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" AdaBoostClassifier training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    # ========================
    # ComplementNB
    # ========================

    @staticmethod
    def ComplementNB(
        dataset_path,
        split_random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        alpha_val=1.0,
        force_alpha_val=True,
        fit_prior_val=True,
        class_prior_val=None,
        norm_val=False,
        save_dir=None,
        save_name="ComplementNB_MN"
    ):
        from sklearn.naive_bayes import ComplementNB

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.ComplementNB_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=split_random_state_val,
                stratify=y,
                shuffle=shuffle_val
            )

            print("Training ComplementNB...")
            model = ComplementNB(
                alpha=alpha_val,
                force_alpha=force_alpha_val,
                fit_prior=fit_prior_val,
                class_prior=class_prior_val,
                norm=norm_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" ComplementNB training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # =========================
    # MLP Classifier
    # =========================

    @staticmethod
    def MLPClassifier(
        dataset_path,
        split_random_state_val=42,
        test_size_val=0.2,
        split_shuffle_val=True,
        hidden_layer_sizes_val=(100,),
        activation_val='relu',
        solver_val='adam',
        alpha_val=0.0001,
        batch_size_val='auto',
        learning_rate_val='constant',
        learning_rate_init_val=0.001,
        power_t_val=0.5,
        max_iter_val=200,
        shuffle_val=True,
        random_state_val=None,
        tol_val=0.0001,
        verbose_val=False,
        warm_start_val=False,
        momentum_val=0.9,
        nesterovs_momentum_val=True,
        early_stopping_val=False,
        validation_fraction_val=0.1,
        beta_1_val=0.9,
        beta_2_val=0.999,
        epsilon_val=1e-08, 
        n_iter_no_change_val=10,
        max_fun_val=15000,
        save_dir=None,
        save_name="MLPClassifier_MN"
    ):
        from sklearn.neural_network import MLPClassifier

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.MLPClassifier_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=split_random_state_val,
                stratify=y,
                shuffle=split_shuffle_val
            )

            print("Training MLPClassifier...")
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes_val,
                activation=activation_val,
                solver=solver_val,
                alpha=alpha_val,
                batch_size=batch_size_val,
                learning_rate=learning_rate_val,
                learning_rate_init=learning_rate_init_val,
                power_t=power_t_val,
                max_iter=max_iter_val,
                shuffle=shuffle_val,
                random_state=random_state_val,
                tol=tol_val,
                verbose=verbose_val,
                warm_start=warm_start_val,
                momentum=momentum_val,
                nesterovs_momentum=nesterovs_momentum_val,
                early_stopping=early_stopping_val,
                validation_fraction=validation_fraction_val,
                beta_1=beta_1_val,
                beta_2=beta_2_val,
                epsilon=epsilon_val,
                n_iter_no_change=n_iter_no_change_val,
                max_fun=max_fun_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" MLPClassifier training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # =========================
    # Radius Neighbors
    # =========================

    @staticmethod
    def RadiusNeighborsClassifier(
        dataset_path,
        split_random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        radius_val=60.0,
        weights_val='distance',
        algorithm_val='auto',
        leaf_size_val=30,
        p_val=2,
        metric_val='minkowski',
        outlier_label_val=-1,
        metric_params_val=None, 
        n_jobs_val=None,
        save_dir=None,
        save_name="RadiusNeighborsClassifier_MN"
    ):
        from sklearn.neighbors import RadiusNeighborsClassifier

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder, scaler = Preprocessing_Scikit_Img_Classif_Supervised_MN.RadiusNeighborsClassifier_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=split_random_state_val,
                stratify=y,
                shuffle=shuffle_val
            )

            print("Training RadiusNeighborsClassifier...")
            model = RadiusNeighborsClassifier(
                radius=radius_val,
                weights=weights_val,
                algorithm=algorithm_val,
                leaf_size=leaf_size_val,
                p=p_val,
                metric=metric_val,
                outlier_label=outlier_label_val,
                metric_params=metric_params_val,
                n_jobs=n_jobs_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder,
                "scaler": scaler
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" RadiusNeighborsClassifier training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    # =========================
    # Extra Tree Classifier
    # =========================

    @staticmethod
    def ExtraTreesClassifier(
        dataset_path,
        split_random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        n_estimators_val = 100,
        criterion_val='gini',
        max_depth_val=None,
        min_samples_split_val=2,
        min_samples_leaf_val=1,
        min_weight_fraction_leaf_val=0.0,
        max_features_val='sqrt',
        max_leaf_nodes_val=None,
        min_impurity_decrease_val=0.0,
        bootstrap_val=False,
        oob_score_val=False,
        n_jobs_val=None,
        random_state_val=None,
        verbose_val=0,
        warm_start_val=False,
        class_weight_val=None,
        ccp_alpha_val=0.0,
        max_samples_val=None,
        monotonic_cst_val=None,
        save_dir=None,
        save_name="ExtraTreesClassifier_MN"
    ):
        from sklearn.ensemble import ExtraTreesClassifier

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder, scaler = Preprocessing_Scikit_Img_Classif_Supervised_MN.ExtraTreeClassifier_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=split_random_state_val,
                stratify=y,
                shuffle=shuffle_val
            )

            print("Training ExtraTreesClassifier...")
            model = ExtraTreesClassifier(
                n_estimators=n_estimators_val,
                criterion=criterion_val,
                max_depth=max_depth_val,
                min_samples_split=min_samples_split_val,
                min_samples_leaf=min_samples_leaf_val,
                min_weight_fraction_leaf=min_weight_fraction_leaf_val,
                max_features=max_features_val,
                max_leaf_nodes=max_leaf_nodes_val,
                min_impurity_decrease=min_impurity_decrease_val,
                bootstrap=bootstrap_val,
                oob_score=oob_score_val,
                n_jobs=n_jobs_val,
                random_state=random_state_val,
                verbose=verbose_val,
                warm_start=warm_start_val,
                class_weight=class_weight_val,
                ccp_alpha=ccp_alpha_val,
                max_samples=max_samples_val,
                monotonic_cst=monotonic_cst_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder,
                "scaler": scaler
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" ExtraTreesClassifier training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # =========================
    # HistGradientBoosting Classifier
    # =========================

    @staticmethod
    def HistGradientBoostingClassifier(
        dataset_path,
        random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        learning_rate_val=0.05,
        max_iter_val=300,
        max_depth_val=8,
        min_samples_leaf_val=10,
        l2_regularization_val=0.1,
        save_dir=None,
        save_name="HistGradientBoostingClassifier_MN"
    ):
        from sklearn.ensemble import HistGradientBoostingClassifier

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.HistGradientBoostingClassifier_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=random_state_val,
                stratify=y,
                shuffle=shuffle_val
            )

            print("Training HistGradientBoostingClassifier...")
            model = HistGradientBoostingClassifier(
                learning_rate=learning_rate_val,
                max_iter=max_iter_val,
                max_depth=max_depth_val,
                min_samples_leaf=min_samples_leaf_val,
                l2_regularization=l2_regularization_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" HistGradientBoostingClassifier training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    # =========================
    # Voting Classifier
    # =========================

    @staticmethod
    def VotingClassifier(
        dataset_path,
        estimators=None,
        random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        voting_val='hard',
        weights_val=None,
        n_jobs_val=None,
        flatten_transform_val=True,
        verbose_val=False,
        save_dir=None,
        save_name="VotingClassifier_MN"
    ):
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.neighbors import KNeighborsClassifier

        try:
            if estimators is None:
                estimators=[
                    ("sgd",SGDClassifier(random_state=random_state_val)),
                    ("rf",RandomForestClassifier(random_state=random_state_val)),
                    ("knn",KNeighborsClassifier())
                ]
            
            print("Loading and preprocessing data...")
            X, y, scaler, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.VotingClassifier_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=random_state_val,
                stratify=y,
                shuffle=shuffle_val
            )

            print("Training VotingClassifier...")
            model = VotingClassifier(
                estimators=estimators,
                voting=voting_val,
                weights=weights_val,
                n_jobs=n_jobs_val,
                flatten_transform=flatten_transform_val,
                verbose=verbose_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "scaler": scaler,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" VotingClassifier training, evaluation, and model saving completed successfully.")

            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    # =========================
    # NearestCentroid Classifier
    # =========================

    @staticmethod
    def NearestCentroid(
        dataset_path,
        random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        metric_val='euclidean',
        shrink_threshold_val=None,
        priors_val='uniform',
        save_dir=None,
        save_name="NearestCentroid_MN"
    ):
        from sklearn.neighbors import NearestCentroid

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.NearestCentroid_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=random_state_val,
                stratify=y,
                shuffle=shuffle_val,   
            )

            print("Training NearestCentroid...")
            model = NearestCentroid(
                metric=metric_val,
                shrink_threshold=shrink_threshold_val,
                priors=priors_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" NearestCentroid training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # =========================
    # Multinomial NB
    # =========================
    @staticmethod
    def MultinomialNB(
        dataset_path,
        random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        alpha_val=1.0,
        fit_prior_val=True,
        class_prior_val=None,
        force_alpha_val=True,
        save_dir=None,
        save_name="MultinomialNB_MN"
    ):
        from sklearn.naive_bayes import MultinomialNB

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.MultinomialNB_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=random_state_val,
                stratify=y,
                shuffle=shuffle_val,   
            )

            print("Training MultinomialNB...")
            model = MultinomialNB(
                alpha=alpha_val,
                fit_prior=fit_prior_val,
                class_prior=class_prior_val,
                force_alpha=force_alpha_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" MultinomialNB training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # =========================
    # Categorical NB
    # =========================
    @staticmethod
    def CategoricalNB(
        dataset_path,
        random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        alpha_val=1.0,
        force_alpha_val=True,
        fit_prior_val=True,
        class_prior_val=None,
        min_categories_val=None,
        save_dir=None,
        save_name="CategoricalNB_MN"
    ):
        from sklearn.naive_bayes import CategoricalNB

        try:
            print("Loading and preprocessing data...")
            X, y, pca, scaler, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.CategoricalNB_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=random_state_val,
                stratify=y,
                shuffle=shuffle_val,   
            )

            print("Training CategoricalNB...")
            model = CategoricalNB(
                alpha=alpha_val,
                fit_prior=fit_prior_val,
                class_prior=class_prior_val,
                force_alpha=force_alpha_val,
                min_categories=min_categories_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "scaler": scaler,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )
            print(" CategoricalNB training, evaluation, and model saving completed successfully.")
            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # =========================
    # Bernoulli NB
    # =========================
    @staticmethod
    def BernoulliNB(
        dataset_path,
        random_state_val=42,
        test_size_val=0.2,
        shuffle_val=True,
        alpha_val=1.0,
        force_alpha_val=True,
        binarize_val=0.0,
        fit_prior_val=True,
        class_prior_val=None,
        save_dir=None,
        save_name="BernoulliNB_MN"
    ):
        from sklearn.naive_bayes import BernoulliNB

        try:
            print("Loading and preprocessing data...")
            X, y, pca, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.BernoulliNB_MN(
                dataset_path=dataset_path,
                mode="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=random_state_val,
                stratify=y,
                shuffle=shuffle_val,   
            )

            print("Training BernoulliNB...")
            model = BernoulliNB(
                alpha=alpha_val,
                force_alpha=force_alpha_val,
                binarize=binarize_val,
                fit_prior=fit_prior_val,
                class_prior=class_prior_val,
                
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle = {
                "model": model,
                "pca": pca,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle, save_name, save_dir
            )

            return bundle

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    # =========================
    # SGDClassifier
    # =========================
    @staticmethod
    def SGDClassifier(
        dataset_path,
        test_size_val=0.2,
        split_random_state_val=42,
        split_shuffle_val=True,
        loss_val='hinge',
        penalty_val='l2',
        alpha_val=0.0001,
        l1_ratio_val=0.15,
        fit_intercept_val=True,
        max_iter_val=1000,
        tol_val=0.001,
        shuffle_val=True,
        verbose_val=0,
        epsilon_val=0.1,
        n_jobs_val=None,
        random_state_val=None,
        learning_rate_val='optimal',
        eta0_val=0.01,
        power_t_val=0.5,
        early_stopping_val=False,
        validation_fraction_val=0.1,
        n_iter_no_change_val=5,
        class_weight_val=None,
        warm_start_val=False,
        average_val=False,
        save_dir=None,
        save_name="SGDClassifier_MN"
    ):
        from sklearn.linear_model import SGDClassifier

        try:
            print("Loading and preprocessing data...")
            X, y, scaler, label_encoder = Preprocessing_Scikit_Img_Classif_Supervised_MN.SGDClassifier_MN(
                dataset_path = dataset_path,
                mode ="train"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size_val,
                random_state=split_random_state_val,
                stratify=y,
                shuffle=split_shuffle_val
            )

            print("Training SGDClassifier...")
            model = SGDClassifier(
                loss=loss_val,
                penalty=penalty_val,
                alpha=alpha_val,
                l1_ratio=l1_ratio_val,
                fit_intercept=fit_intercept_val,
                max_iter=max_iter_val,
                tol=tol_val,
                shuffle=shuffle_val,
                verbose=verbose_val,
                epsilon=epsilon_val,
                n_jobs=n_jobs_val,
                random_state=random_state_val,
                learning_rate=learning_rate_val,
                eta0=eta0_val,
                power_t=power_t_val,
                early_stopping=early_stopping_val,
                validation_fraction=validation_fraction_val,
                n_iter_no_change=n_iter_no_change_val,
                class_weight=class_weight_val,
                warm_start=warm_start_val,
                average=average_val
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            Scikit_Img_Classif_Supervised._evaluate(y_test, y_pred)

            bundle={
                "model": model,
                "scaler": scaler,
                "label_encoder": label_encoder
            }

            Scikit_Img_Classif_Supervised._save_bundle(
                bundle=bundle, prefix=save_name, save_dir=save_dir
            )
            print(" SGDClassifier training, evaluation, and model saving completed successfully.")         
            return bundle
        except Exception as e:
            print(f"An error occured:{e}")
            return None