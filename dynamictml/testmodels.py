class test_sci_img_classification_supervised:

    def test_NuSVC_Classifier(bundle,img_path):
        model = bundle["model"]
        pca = bundle["pca"][0]
        scaler = bundle["pca"][1]
        label_encoder = bundle["label_encoder"]

        from dynamictml.dataset_loader_g import Preprocessing_Scikit_Img_Classif_Supervised_MN

        x_pred = Preprocessing_Scikit_Img_Classif_Supervised_MN.svc_MN(
            img_paths=[img_path],
            pca=pca,
            scaler=scaler,
            mode="predict"
        )
        y_pred_encoded = model.predict(x_pred)
        y_pred_label = label_encoder.inverse_transform(y_pred_encoded)
        print("Predicted class:", y_pred_label[0])
        return y_pred_label[0]

    def test_KNeighbors_Classifier(bundle,img_path):
        model = bundle["model"]
        pcs = bundle["pca"][0] 
        scaler = bundle["pca"][1]
        label_encoder = bundle["label_encoder"]

        from dynamictml.dataset_loader_g import Preprocessing_Scikit_Img_Classif_Supervised_MN

        x_pred = Preprocessing_Scikit_Img_Classif_Supervised_MN.knn_MN(
            img_paths=[img_path],
            pca=pcs,
            scaler=scaler,
            mode = "predict"
        )
        y_pred_endcoded = model.predict(x_pred)
        y_pred_label = label_encoder.inverse_transform(y_pred_endcoded)
        print("Predicted class:",y_pred_label[0])
        return y_pred_label[0]

        
    def test_GaussianProcess_MN(bundle,img_path):
        model = bundle["model"]
        pca = bundle["pca"]
        label_encoder = bundle["label_encoder"]

        from dynamictml.dataset_loader_g import Preprocessing_Scikit_Img_Classif_Supervised_MN


        X_pred = Preprocessing_Scikit_Img_Classif_Supervised_MN.gaussianprocess_MN(
            img_paths=[img_path],
            pca=pca,
            mode="predict"
        )


        y_pred_encoded = model.predict(X_pred)
        y_pred_label = label_encoder.inverse_transform(y_pred_encoded)

        print("Predicted class:", y_pred_label[0])
        return y_pred_label[0]
    


